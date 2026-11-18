async (page) => {
  const dialog = page.getByRole('dialog', {name:'实验配置',exact:true});
  if (!(await dialog.count())) await page.getByRole('button', {name:'实验配置',exact:true}).click();
  await dialog.getByRole('spinbutton', {name:'训练步数 steps · 1–200',exact:true}).fill('200');
  const submitted=page.waitForResponse(response => response.url().endsWith('/api/runs') && response.request().method()==='POST');
  await dialog.getByRole('button', {name:'运行实验',exact:true}).click();
  const response=await submitted;
  const record=await response.json();
  if (!response.ok() || !record.run_id) throw new Error('Run submission failed: '+JSON.stringify(record));
  await page.getByRole('button', {name:'停止实验',exact:true}).click();
  await page.getByText(new RegExp('实验已取消 · r')).waitFor({timeout:20000});
  await page.getByRole('button', {name:'运行实验',exact:true}).waitFor();
  await page.screenshot({path:'../../output/playwright/studio-cancelled.png',scale:'css'});
  await page.getByRole('button', {name:'实验配置',exact:true}).click();
  await dialog.getByRole('spinbutton', {name:'训练步数 steps · 1–200',exact:true}).fill('20');
  await dialog.getByRole('button', {name:'完成配置',exact:true}).click();
  return {run_id:record.run_id,observed:'实验已取消',steps_restored:20};
}
