export async function api(path, body) {
  let response;
  try {
    response = await fetch(`/api${path}`, body === undefined ? undefined : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
  } catch {
    throw new Error('无法连接本地计算服务。画布草稿仍保存在此浏览器中。');
  }
  let result;
  try { result = await response.json(); }
  catch { throw new Error(`本地服务未就绪（HTTP ${response.status}），请检查服务状态。`); }
  if (!response.ok) throw new Error(result.error || `请求失败（HTTP ${response.status}）`);
  return result;
}
export function download(name, content, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
