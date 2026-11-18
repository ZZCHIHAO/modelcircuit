async (page) => {
  const notes=[];
  const check=(ok,message)=>{if(!ok)throw new Error(message);notes.push(message);};
  const out='../../output/playwright/';
  // Playwright CLI tracks file choosers separately; set the known hidden
  // JSON input directly so a chooser cannot outlive a subsequent page reload.
  await page.locator('input[type="file"][accept=".json,application/json"]').setInputFiles(out+'studio-edited-project.json');
  await page.getByRole('button',{name:'共享编码器 ×2',exact:true}).click();
  check(await page.getByRole('spinbutton',{name:'特征维度 hidden_features',exact:true}).inputValue()==='10','Downloaded JSON restores the source-edited hidden dimension');
  await page.reload();
  await page.getByRole('button',{name:'共享编码器 ×2',exact:true}).click();
  check(await page.getByRole('spinbutton',{name:'特征维度 hidden_features',exact:true}).inputValue()==='10','The project draft persists across a browser reload');
  await page.getByRole('button',{name:'模型',exact:true}).click();
  await page.getByRole('button',{name:'载入双路分类器示例',exact:true}).click();
  await page.getByText('结构有效',{exact:true}).waitFor();
  await page.waitForFunction(()=>document.querySelectorAll('.react-flow__node').length===8);
  await page.getByRole('button',{name:'共享编码器 ×2',exact:true}).click();
  check(await page.getByRole('spinbutton',{name:'特征维度 hidden_features',exact:true}).inputValue()==='8','The default example is restored for handoff');
  for (const [width,height] of [[1100,900],[760,900],[1440,1024]]) {
    await page.setViewportSize({width,height});
    await page.getByRole('button',{name:'适应画布',exact:true}).click();
    await page.waitForFunction(()=>{const c=document.querySelector('.react-flow__viewport');return c && c.style.transform.includes('translate');});
    const layout=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,controls:[...document.querySelectorAll('.app-header button,.file-actions button')].filter(x=>x.getBoundingClientRect().width>0).map(x=>({name:x.getAttribute('aria-label')||x.textContent,rect:{x:x.getBoundingClientRect().x,right:x.getBoundingClientRect().right}}))}));
    check(layout.scroll<=layout.width+1,`${width}px viewport does not overflow horizontally`);
    check(layout.controls.every(x=>x.rect.x>=0&&x.rect.right<=width+1),`${width}px persistent header and file controls stay inside the viewport`);
    await page.screenshot({path:out+`studio-layout-${width}.png`,scale:'css'});
  }
  await page.getByRole('button',{name:'Python',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Python 与画布',exact:true});
  await page.waitForFunction(()=>{const field=document.querySelector('textarea[aria-label="Python 模型代码"]');return field&&!field.readOnly&&field.value.includes('class Model');});
  await page.screenshot({path:out+'studio-python.png',scale:'css'});
  await dialog.getByRole('button',{name:'关闭',exact:true}).click();
  return notes;
}
