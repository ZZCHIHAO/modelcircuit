"""Exact standalone v4 helpers and literal policy parsing (no torch import)."""
import ast

VISION_KINDS = frozenset({'BatchNorm2d', 'Identity', 'Sigmoid', 'Multiply', 'Scale', 'OneMinus'})
HELPER_KINDS = ('Multiply', 'Scale', 'OneMinus', 'BatchNorm2d')

HELPERS = '''class Multiply(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        if a.shape != b.shape:
            raise ValueError("Multiply requires equal runtime shapes; broadcasting is not implicit")
        return a * b


class Scale(nn.Module):
    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = float(factor)

    def forward(self, input):
        return self.factor * input


class OneMinus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return 1 - input


class BatchNorm2d(nn.BatchNorm2d):
    def forward(self, input):
        self._check_input_dim(input)
        if input.shape[1] != self.num_features:
            raise ValueError("BatchNorm2d input channels do not match num_features")
        uses_batch_statistics = self.training or (self.running_mean is None and self.running_var is None)
        if uses_batch_statistics and input.shape[0] * input.shape[2] * input.shape[3] <= 1:
            raise ValueError("BatchNorm2d batch statistics require more than 1 value per channel (N*H*W > 1)")
        return super().forward(input)
'''

POLICY_HELPER = '''def _check_execution_policy(*inputs):
    for input in inputs:
        if input.ndim < 1 or not 1 <= input.shape[0] <= MLFORGE_EXECUTION_POLICY['resource_batch_bound']:
            raise ValueError("Input batch must be between 1 and resource_batch_bound")
'''


def parse_policy(node):
    """Require a literal with unique, exact keys; ast.literal_eval loses duplicates."""
    if not isinstance(node, ast.Dict) or len(node.keys) not in (1, 2, 3):
        raise ValueError('MLFORGE_EXECUTION_POLICY must be an exact literal policy dictionary')
    try:
        keys = [ast.literal_eval(key) for key in node.keys]
        if any(type(key) is not str for key in keys) or len(set(keys)) != len(keys) or set(keys) not in ({'resource_profile'}, {'resource_profile', 'resource_batch_bound'}, {'resource_profile', 'resource_batch_bound', 'capability_profile'}):
            raise ValueError('Invalid policy keys')
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        raise ValueError('MLFORGE_EXECUTION_POLICY requires literal resource_profile/resource_batch_bound') from None
    if value['resource_profile'] not in (None, 'classification-v2', 'vision-v4', 'hardware-v5'):
        raise ValueError('Unsupported MLFORGE_EXECUTION_POLICY resource_profile')
    if 'capability_profile' in value and (value['capability_profile'] != 'tensor-v1' or value['resource_profile'] != 'hardware-v5'):
        raise ValueError('Unsupported tensor execution policy')
    from .schema import graph_resource_limits
    graph_resource_limits(value)
    return value


def exact_policy_helper(node):
    expected = ast.parse(POLICY_HELPER).body[0]
    return ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False)


def remove_policy_guard(node, policy, helper_present):
    """Validate/strip one exact entry guard; the caller verifies its root scope."""
    forward = next((child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == 'forward'), None)
    if forward is None:
        return False
    guarded = False
    for statement in list(forward.body):
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call) or not isinstance(statement.value.func, ast.Name) or statement.value.func.id != '_check_execution_policy':
            continue
        expression = statement.value
        names = [argument.arg for argument in forward.args.args[1:]]
        # Python determines local names over the entire function, so even an
        # assignment after the first guard makes that earlier call unbound.
        if any(isinstance(reference, ast.Name) and reference.id == '_check_execution_policy' and
               isinstance(reference.ctx, (ast.Store, ast.Del)) for reference in ast.walk(forward)):
            raise ValueError('Local binding shadows the active execution policy guard')
        if guarded or policy is None or policy['resource_profile'] not in ('vision-v4', 'hardware-v5') or not helper_present or statement is not forward.body[0] or '_check_execution_policy' in names or expression.keywords or len(expression.args) != len(names) or any(not isinstance(value, ast.Name) or value.id != name for value, name in zip(expression.args, names)):
            raise ValueError('Unsupported execution policy helper invocation')
        forward.body.remove(statement)
        guarded = True
    return guarded
