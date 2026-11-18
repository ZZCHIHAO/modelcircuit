async (page) => {
  const out='../../output/playwright/';
  await page.getByRole('button',{name:'训练曲线',exact:true}).click();
  await page.getByRole('img',{name:/实测训练损失曲线/}).waitFor();
  await page.screenshot({path:out+'studio-final-root.png',scale:'css'});
  await page.getByRole('button',{name:'模型参数',exact:true}).click();
  const rows=page.locator('.data-table-wrap tbody tr');
  if (await rows.count()!==4) throw new Error('The shared default model must expose four parameter tensors');
  const values=await rows.allInnerTexts();
  if (!values.every(x=>/0\./.test(x))) throw new Error('Measured gradient/update values missing');
  await page.screenshot({path:out+'studio-final-parameters.png',scale:'css'});
  await page.setViewportSize({width:760,height:900});
  const hidden=page.getByRole('spinbutton',{name:'特征维度 hidden_features',exact:true});
  await hidden.scrollIntoViewIfNeeded();
  const rect=await hidden.boundingBox();
  if (!rect || rect.x<0 || rect.x+rect.width>760) throw new Error('Narrow inspector is unreachable');
  await page.screenshot({path:out+'studio-narrow-inspector.png',scale:'css'});
  await page.setViewportSize({width:1440,height:1024});
  await page.evaluate(()=>scrollTo(0,0));
  await page.getByRole('button',{name:'encode_b · 内部结构',exact:true}).click();
  await page.locator('.react-flow__node[data-id="call:project"]').click();
  await page.getByRole('button',{name:'探针数据',exact:true}).click();
  await page.locator('.probe-sidebar code').filter({hasText:'call:encode_b/call:project'}).waitFor();
  await page.screenshot({path:out+'studio-final-chip.png',scale:'css'});
  return {parameter_rows:values,narrow_inspector_reachable:true,final_probe:'call:encode_b/call:project'};
}
