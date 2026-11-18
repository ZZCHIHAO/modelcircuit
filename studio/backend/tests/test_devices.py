import unittest
from studio_core.devices import device_capabilities, resolve_device


class DeviceTests(unittest.TestCase):
    def test_capabilities_separate_runtime_and_hardware(self):
        result=device_capabilities()
        self.assertIn('torch_version',result)
        self.assertIn('hardware_gpus',result)
        self.assertFalse(result['multi_gpu_supported'])
        self.assertTrue(any(item['id']=='cpu' for item in result['devices']))
        if not result['cuda_available']:
            self.assertTrue(result['reason'])
            self.assertFalse(any(item['id'].startswith('cuda:') for item in result['devices']))

    def test_explicit_devices_never_fall_back(self):
        self.assertEqual(str(resolve_device('cpu')),'cpu')
        for requested in ('auto','cuda','gpu:0','cuda:-1','cpu:0'):
            with self.subTest(requested=requested),self.assertRaises(ValueError):resolve_device(requested)
        with self.assertRaises(ValueError):resolve_device('cuda:99999')


if __name__=='__main__':unittest.main()
