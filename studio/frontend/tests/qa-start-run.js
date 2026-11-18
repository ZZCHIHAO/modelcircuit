async (page) => {
  const submitted=page.waitForResponse(response => response.url().endsWith('/api/runs') && response.request().method()==='POST');
  await page.getByRole('button', {name:'运行实验',exact:true}).click();
  const response=await submitted;
  const run=await response.json();
  if (!response.ok()) throw new Error(JSON.stringify(run));
  await page.getByText(new RegExp('实验已完成 · r')).waitFor({timeout:30000});
  await page.getByRole('button', {name:'探针数据',exact:true}).click();
  await page.locator('.probe-metadata').waitFor();
  return {run_id:run.run_id,revision:run.revision,observed:'实验已完成'};
}
