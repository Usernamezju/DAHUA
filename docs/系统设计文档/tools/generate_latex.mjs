import fs from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const source = JSON.parse(await fs.readFile(path.join(root, 'source/notion-export.json'), 'utf8'));
const chaptersDir = path.join(root, 'chapters');
const figuresDir = path.join(root, 'figures');
const stylesDir = path.join(root, 'styles');
await Promise.all([chaptersDir, figuresDir, stylesDir].map((dir) => fs.mkdir(dir, { recursive: true })));

const chapterFiles = [
  '01_overview', '02_requirements', '03_architecture', '04_pose_features', '05_protogcn',
  '06_llm_teacher', '07_collaboration', '08_distillation', '09_incremental_learning',
  '10_web_visualization', '11_code_structure', '12_datasets', '13_deployment',
  '14_completion', '15_references',
];

const imageNames = new Map();
let imageCounter = 0;
function richText(items = []) { return items.map((item) => item.plain_text ?? item.text?.content ?? '').join(''); }
function content(block) {
  const data = block[block.type] ?? {};
  if (typeof data.title === 'string') return data.title;
  return richText(data.rich_text ?? data.title ?? []);
}
function stripNumber(title) { return title.replace(/^\s*\d+(?:\.\d+)*\.?\s+/, '').trim(); }
function escapeText(value) {
  return value
    .replace(/\\/g, '\\textbackslash{}')
    .replace(/([{}#$%&_])/g, '\\$1')
    .replace(/~/g, '\\textasciitilde{}')
    .replace(/\^/g, '\\textasciicircum{}');
}
function normalizeMath(value) {
  return value
    .replace(/[\u2061\u200a]/g, '')
    .replace(/−/g, '-')
    .replace(/≤/g, '\\leq ')
    .replace(/≥/g, '\\geq ')
    .replace(/∈/g, '\\in ')
    .replace(/∥/g, '\\lVert ')
    .replace(/∣/g, '|')
    .replace(/∩/g, '\\cap ')
    .replace(/∪/g, '\\cup ')
    .replace(/⋅/g, '\\cdot ')
    .replace(/⊤/g, '^{\\top}')
    .replace(/→/g, '\\rightarrow ')
    .replace(/↔/g, '\\leftrightarrow ')
    .replace(/\\textbf/g, '\\text');
}
function extractFormula(value) {
  const normalized = value.replace(/[\u2061\u200a]/g, '').trim();
  // Preserve complete equations that begin with a conventional symbolic
  // assignment; their Notion visual source may otherwise be duplicated.
  if (/^(?:D_\{inc\}|\|D_\{inc\}\||\\Delta\s+W|W')\s*=/.test(normalized)) {
    return normalizeMath(normalized);
  }
  const repeatedAssignment = normalized.match(/^([A-Za-z])\s*=\s*([\d.]+)\1\s*=\s*([\d.]+)$/);
  if (repeatedAssignment) return `${repeatedAssignment[1]}=${repeatedAssignment[3]}`;
  const namedAssignment = normalized.match(/([A-Za-z]_\{\\(?:mathrm|operatorname)\{[^}]+\}\}(?:\([^)]*\))?\s*=[\s\S]*)/);
  if (namedAssignment) return normalizeMath(namedAssignment[1]);
  const operatorAssignment = normalized.match(/([A-Za-z](?:_\{[^}]+\}|_[A-Za-z0-9])?(?:\^(?:\{[^}]+\}|[A-Za-z0-9]))?\s*=\s*\\(?:arg|max|min|operatorname)[\s\S]*)/);
  if (operatorAssignment) return normalizeMath(operatorAssignment[1]);
  // Prefer the final LaTeX expression when Notion prefixes it with a visual
  // duplicate, e.g. "It∈...I_t\\in...".
  const arrowFormula = normalized.match(/([A-Za-z](?:_\{[^}]+\}|_[A-Za-z0-9])?(?:\^(?:\{[^}]+\}|[A-Za-z0-9]))?)\s*\\xrightarrow/);
  if (arrowFormula) {
    let expression = normalized.slice(arrowFormula.index).trim();
    if (normalized.slice(0, arrowFormula.index).includes('\\boxed')) expression = expression.replace(/\n\}\s*$/, '');
    return normalizeMath(expression);
  }
  const tailFormula = [...normalized.matchAll(/(?<![A-Za-z])(?:\\(?:mathbf|mathcal)\s*[A-Za-z]|[A-Za-z](?:_\{[^}]+\}|_[A-Za-z0-9])?(?:\^(?:\{[^}]+\}|[A-Za-z0-9]))?)\s*\\(?:in|x?rightarrow|leftrightarrow|not\\rightarrow)/g)][0];
  if (tailFormula) {
    let expression = normalized.slice(tailFormula.index).trim();
    if (normalized.slice(0, tailFormula.index).includes('\\boxed')) expression = expression.replace(/\n\}\s*$/, '');
    return normalizeMath(expression);
  }
  const caseAssignment = [...normalized.matchAll(/(?:Mask_\{[^}]+\}(?:\^(?:\{[^}]+\}|[A-Za-z0-9]))?)\s*=\s*\\begin\{cases\}/g)].at(-1)
    ?? [...normalized.matchAll(/(?<![A-Za-z])(?:[A-Za-z]+(?:_\{[^}]+\}|_[A-Za-z0-9])?(?:\^(?:\{[^}]+\}|[A-Za-z0-9]))?)\s*=\s*\\begin\{cases\}/g)].at(-1);
  if (caseAssignment) return normalizeMath(normalized.slice(caseAssignment.index).trim());
  const modulo = [...normalized.matchAll(/(?<![A-Za-z])([A-Za-z])\s*\\bmod\s*[A-Za-z]\s*=\s*\d+/g)].at(-1);
  if (modulo) return normalizeMath(modulo[0]);
  const greekAssignment = [...normalized.matchAll(/\\tau_\{[^}]+\}\s*=\s*[\d.]+/g)].at(-1);
  if (greekAssignment) return normalizeMath(greekAssignment[0]);
  const comparison = normalized.match(/^([A-Za-z])([≤≥<>])[\s\S]*?(\\(?:leq|geq)|[A-Za-z]_(?:\{[^}]+\}|[A-Za-z0-9]))/);
  if (comparison) {
    const [, left, operator, right] = comparison;
    const renderedOperator = normalizeMath(operator);
    if (right.startsWith('\\')) return `${left}${normalizeMath(normalized.slice(normalized.indexOf(right))).trim()}`;
    return `${left}${renderedOperator}${normalizeMath(right)}`;
  }
  const commandIndex = normalized.search(/\\(?:[A-Za-z]+|[{}])/);
  if (commandIndex >= 0) {
    const prefix = normalized.slice(0, commandIndex);
    const command = normalized.slice(commandIndex).match(/^\\[A-Za-z]+/)?.[0] ?? '';
    if (['\\leq', '\\geq', '\\times'].includes(command)) {
      const word = prefix.match(/([A-Za-z]+)$/)?.[1];
      const digit = prefix.match(/(\d)$/)?.[1];
      if (word || digit) return `${word ?? digit}${normalizeMath(normalized.slice(commandIndex)).trim()}`;
    }
    const nestedSubscript = prefix.match(/[A-Za-z]_\{$/);
    if (nestedSubscript && (command !== '\\text' || !prefix.includes('='))) return normalizeMath(normalized.slice(commandIndex - nestedSubscript[0].length).trim());
    const assignments = [...prefix.matchAll(/[A-Za-z](?:_\{[^}]+\}|_[A-Za-z0-9])?(?:\^(?:\{[^}]+\}|[A-Za-z0-9]))?\s*=\s*(?:[A-Za-z](?:_\{[^}]+\}|_[A-Za-z0-9])?(?:\^(?:\{[^}]+\}|[A-Za-z0-9]))?\s*)*/g)];
    const assignment = assignments.at(-1);
    const assignmentTail = assignment ? prefix.slice(assignment.index + assignment[0].length) : '';
    if (assignment && commandIndex - assignment.index < 90 && (command === '\\text' || (!/[-−+()]/.test(assignmentTail) && !/[{}]/.test(prefix.slice(assignment.index))))) {
      return normalizeMath(normalized.slice(assignment.index).trim());
    }
    return normalizeMath(normalized.slice(commandIndex).trim());
  }
  const candidates = [
    ...normalized.matchAll(/(?:[A-Za-z](?:_\{[^}]+\}|_[A-Za-z0-9])|\\[A-Za-z]+)/g),
  ];
  if (!candidates.length) return null;
  return normalizeMath(normalized.slice(candidates[0].index).trim());
}
function inlineMath(value) {
  const slots = [];
  const slot = (math) => {
    const marker = `@@MATH${slots.length}@@`;
    slots.push(`\\(${normalizeMath(math)}\\)`);
    return marker;
  };
  let text = normalizeMath(value);
  text = text.replace(/\\\((.*?)\\\)/g, (_full, math) => slot(math));
  text = text.replace(/([A-Za-z][A-Za-z0-9,]*)(\\(?:mathbf|mathcal)\s*(?:\{[^}]+\}|[A-Za-z]))/g,
    (_full, _duplicate, math) => slot(math));
  text = text.replace(/[A-Za-z]ˉ[A-Za-z](\\bar\{[A-Za-z]\}_[A-Za-z])/g,
    (_full, math) => slot(math));
  text = text.replace(/([A-Za-z][A-Za-z0-9,]*)([A-Za-z]_\{\\(?:text|mathrm|operatorname)\{[^}]+\}\}(?:\^(?:\{[^}]+\}|[A-Za-z0-9]))?)/g,
    (_full, _duplicate, math) => slot(math));
  text = text.replace(/(?<![A-Za-z0-9])([A-Za-z]_\{\\(?:text|mathrm|operatorname)\{[^}]+\}\}(?:\^(?:\{[^}]+\}|[A-Za-z0-9]))?)(?![A-Za-z0-9])/g,
    (_full, math) => slot(math));
  text = text.replace(/([A-Za-z][A-Za-z0-9,]*)([A-Za-z]_(?:\{[^}]+\}|[A-Za-z0-9])(?:\^(?:\{[^}]+\}|[A-Za-z0-9]))?)/g,
    (_full, _duplicate, math) => slot(math));
  text = text.replace(/(?<![A-Za-z0-9])([A-Za-z]_(?:\{[^}]+\}|[A-Za-z0-9])(?:\^(?:\{[^}]+\}|[A-Za-z0-9]))?)(?![A-Za-z0-9])/g,
    (_full, math) => slot(math));
  text = text.replace(/第\s*([A-Za-z])\1\s*(帧|个|类|条)/g, (_full, variable, unit) => `第 ${slot(variable)} ${unit}`);
  text = text.replace(/([A-Z])\1(?=\s*表示)/g, (_full, variable) => slot(variable));
  text = escapeText(text);
  return text.replace(/@@MATH(\d+)@@/g, (_full, index) => slots[Number(index)]);
}
function flowBlock(value) {
  const steps = value.split(/(?:\r?\n|→)/).map((step) => step.trim()).filter(Boolean);
  if (steps.length < 2) return null;
  return `\\[
\\begin{gathered}
${steps.map((step, index) => `\\text{${escapeText(step)}}${index < steps.length - 1 ? '\\\\[-0.2em]\n\\downarrow\\\\[-0.2em]' : ''}`).join('\n')}
\\end{gathered}
\\]
\n`;
}
function textBlock(value) {
  const normalized = value.trim();
  if (!normalized) return '';
  if (normalized === 'Maskm,t,j=[Sm,t,j≥0.20]') {
    return '\\[\n\\mathrm{Mask}_{m,t,j}=[S_{m,t,j}\\geq 0.20]\n\\]\n\n';
  }
  // Notion stores simple process diagrams as plain-text arrows.  Render them
  // as a compact vertical flow rather than escaping the arrow commands.  A
  // LaTeX command indicates that the block is an actual mathematical formula
  // and must keep its original math rendering path.
  if (/→/.test(normalized) && !/\\/.test(normalized)) {
    const flow = flowBlock(normalized);
    if (flow) return flow;
  }
  const formula = extractFormula(normalized);
  const chineseCount = (normalized.match(/[\u4e00-\u9fff]/g) ?? []).length;
  const display = formula && (chineseCount === 0 || /\\(?:boxed|begin\{(?:aligned|cases))/i.test(normalized) || (normalized.match(/\\text\{/g) ?? []).length >= 2);
  if (display) return `\\[\n${formula}\n\\]\n\n`;
  return `${inlineMath(normalized)}\n\n`;
}
function tableCell(value) { return escapeText(value.replace(/\n/g, ' ').trim()); }
function imagePath(block) {
  if (!imageNames.has(block.id)) imageNames.set(block.id, `notion_image_${String(++imageCounter).padStart(2, '0')}.png`);
  return imageNames.get(block.id);
}
function tableSpec(columns) {
  const widths = { 1: '0.94', 2: '0.46', 3: '0.30', 4: '0.22', 5: '0.18', 6: '0.15' };
  const width = widths[columns] ?? (0.9 / columns).toFixed(3);
  return Array.from({ length: columns }, () => `>{\\raggedright\\arraybackslash}p{${width}\\linewidth}`).join('|');
}
function renderTable(block) {
  const rows = block.children ?? [];
  const columns = block.table.table_width;
  const output = [];
  output.push('\\begin{center}', '\\small', `\\begin{longtable}{|${tableSpec(columns)}|}`, '\\hline');
  for (const row of rows) {
    output.push(row.table_row.cells.map((cell) => tableCell(richText(cell))).join(' & ') + ' \\\\', '\\hline');
  }
  output.push('\\end{longtable}', '\\end{center}');
  return output.join('\n') + '\n\n';
}
function renderCode(block) {
  const language = block.code.language || 'plain text';
  const code = richText(block.code.rich_text).replace(/\\end\{NotionCode\}/g, '\\textbackslash{}end{NotionCode}');
  return `\\begin{NotionCode}[${escapeText(language)}]\n${code}\n\\end{NotionCode}\n\n`;
}
function renderBlocks(blocks, headingShift) {
  let output = '';
  for (let index = 0; index < blocks.length; index += 1) {
    const block = blocks[index];
    // The Notion export occasionally places a formula inside standalone "["
    // and "]" paragraphs.  Convert the three blocks back into one display.
    if (block.type === 'paragraph' && content(block).trim() === '[') {
      const middle = blocks[index + 1];
      const closing = blocks[index + 2];
      if (middle?.type === 'paragraph' && closing?.type === 'paragraph' && content(closing).trim() === ']') {
        const expression = extractFormula(content(middle).trim());
        if (expression) {
          output += `\\[\n${expression}\n\\]\n\n`;
          index += 2;
          continue;
        }
      }
    }
    if (block.type === 'bulleted_list_item' || block.type === 'numbered_list_item') {
      const type = block.type;
      const environment = type === 'bulleted_list_item' ? 'itemize' : 'enumerate';
      output += `\\begin{${environment}}\n`;
      while (index < blocks.length && blocks[index].type === type) {
        const item = blocks[index];
        let itemText = content(item).trim();
        const variableChildren = (item.children ?? []).filter((child) => child.type === 'paragraph')
          .map(content).map((text) => text.trim()).filter((text) => /^([A-Za-z])\1$/.test(text));
        if (variableChildren.length === 1) {
          const variable = variableChildren[0][0];
          itemText = itemText.replace(/第\s*个/, `第 \\(${variable}\\) 个`).replace(/第\s*帧/, `第 \\(${variable}\\) 帧`);
        }
        output += `\\item ${inlineMath(itemText)}\n`;
        if (item.children?.length && !variableChildren.length) output += renderBlocks(item.children, headingShift);
        index += 1;
      }
      output += `\\end{${environment}}\n\n`;
      index -= 1;
      continue;
    }
    if (block.type.startsWith('heading_')) {
      const actualLevel = Number(block.type.slice(-1));
      const relativeLevel = Math.max(1, actualLevel - headingShift);
      const command = ['section', 'subsection', 'subsubsection'][Math.min(relativeLevel, 3) - 1];
      output += `\\${command}{${escapeText(stripNumber(content(block)))}}\n\n`;
    } else if (block.type === 'paragraph') {
      output += textBlock(content(block));
    } else if (block.type === 'quote') {
      output += `\\begin{quote}\n${textBlock(content(block))}\\end{quote}\n\n`;
    } else if (block.type === 'code') {
      output += renderCode(block);
    } else if (block.type === 'divider') {
      // Notion dividers are structural markers only; they should not appear
      // as horizontal rules between adjacent document sections.
      output += '\n';
    } else if (block.type === 'equation') {
      output += `\\[\n${block.equation.expression}\n\\]\n\n`;
    } else if (block.type === 'table') {
      output += renderTable(block);
    } else if (block.type === 'image') {
      const figure = imagePath(block);
      const caption = richText(block.image.caption);
      output += `\\begin{figure}[htbp]\n\\centering\n\\includegraphics[width=0.96\\linewidth,keepaspectratio]{figures/${figure}}\n${caption ? `\\caption{${escapeText(caption)}}\n` : ''}\\end{figure}\n\n`;
    } else if (block.type === 'embed') {
      const url = block.embed.url ?? '';
      if (url) output += `\\url{${url}}\n\n`;
    }
    if (block.children?.length && !['table', 'image'].includes(block.type)) output += renderBlocks(block.children, headingShift);
  }
  return output;
}

for (const [index, page] of source.blocks.filter((block) => block.type === 'child_page').entries()) {
  const title = stripNumber(page.child_page.title);
  const headings = (page.children ?? []).filter((block) => block.type.startsWith('heading_')).map((block) => Number(block.type.slice(-1)));
  const headingShift = (headings.length ? Math.min(...headings) : 2) - 1;
  let tex = index === chapterFiles.length - 1
    ? `% Source: Notion page “${page.child_page.title}”. IEEE references supplied by the project team.\n% \\bibliography creates the only visible “参考文献” title.\n\\addcontentsline{toc}{chapter}{参考文献}\n\\nocite{*}\n\\bibliographystyle{ieeetr}\n\\bibliography{bibliography}\n`
    : `% Source: Notion page “${page.child_page.title}”. Generated without rewriting its content.\n\\chapter{${escapeText(title)}}\n\n${renderBlocks(page.children ?? [], headingShift)}`;
  if (chapterFiles[index] === '11_code_structure') {
    tex = tex.replace(/\nGithub仓库：\s*$/, '\n');
    tex += '\\noindent Github仓库：\\url{https://github.com/Usernamezju/DAHUA}\n';
  }
  await fs.writeFile(path.join(chaptersDir, `${chapterFiles[index]}.tex`), tex);
}

const style = String.raw`\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{zju-system-design}[2026/08/31 Zhejiang University system design document]
\RequirePackage[a4paper,top=2.7cm,bottom=2.7cm,left=2.7cm,right=2.7cm,headheight=15pt]{geometry}
\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb}
\RequirePackage{xcolor}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{setspace}
\RequirePackage{enumitem}
\RequirePackage{longtable}
\RequirePackage{array}
\RequirePackage[most]{tcolorbox}
\RequirePackage{hyperref}
\setmonofont{Noto Sans Mono CJK SC}
\definecolor{ZJUPurple}{RGB}{78,61,125}
\hypersetup{colorlinks=true,linkcolor=ZJUPurple,urlcolor=ZJUPurple,citecolor=ZJUPurple}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.35em}
\onehalfspacing
\setlist{nosep,leftmargin=2.4em}
\pagestyle{fancy}
\setlength{\headheight}{25pt}
\fancyhf{}
\fancyhead[L]{\songti 浙江大学}
\fancyhead[R]{\songti 校园行为识别系统设计文档}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{ZJUPurple}\leaders\hrule height \headrulewidth\hfill}}
\ctexset{
  chapter={format=\centering\zihao{2}\songti\color{ZJUPurple},name={第,章},number=\arabic{chapter},beforeskip=1.5em,afterskip=1.5em},
  section={format=\zihao{3}\songti\color{ZJUPurple},number=\arabic{chapter}.\arabic{section},beforeskip=1.3em,afterskip=0.8em},
  subsection={format=\zihao{4}\songti\color{ZJUPurple},number=\arabic{chapter}.\arabic{section}.\arabic{subsection},beforeskip=1.1em,afterskip=0.6em},
  subsubsection={format=\zihao{-4}\songti\color{ZJUPurple},beforeskip=0.9em,afterskip=0.4em}
}
\newtcblisting{NotionCode}[1][]{
  enhanced, breakable,
  colback=ZJUPurple!3, colframe=ZJUPurple!65!black,
  boxrule=0.45pt, arc=1.5mm,
  left=2.5mm, right=2.5mm, top=1.5mm, bottom=1.5mm,
  title={代码片段（#1）}, fonttitle=\small\sffamily,
  coltitle=black, colbacktitle=ZJUPurple!14,
  listing only, listing engine=listings,
  listing options={basicstyle=\small\ttfamily,breaklines=true,columns=fullflexible,showstringspaces=false,keepspaces=true}
}
\newcommand{\ZJUCover}{%
\begin{titlepage}\thispagestyle{empty}\centering
\vspace*{1.5cm}
\IfFileExists{figures/zju_logo.png}{\includegraphics[width=0.27\textwidth]{figures/zju_logo.png}\par}{\color{ZJUPurple}\zihao{1}\songti 浙江大学\par}
\vspace{1.0cm}
{\color{ZJUPurple}\zihao{0}\songti 浙江大学}\par
\vspace{0.35cm}
{\color{ZJUPurple}\Large ZHEJIANG UNIVERSITY}\par
\vfill
{\color{ZJUPurple}\zihao{1}\songti 校园行为识别系统}\par
\vspace{0.45cm}
{\zihao{1}\songti 系统设计文档}\par
\vspace{1.2cm}
\rule{0.72\textwidth}{0.8pt}\par
\vspace{0.7cm}
{\large\songti 大华杯项目设计文档}\par
\vfill
{\large\songti \today}\par
\vspace*{1.2cm}
\end{titlepage}%
}
`;
await fs.writeFile(path.join(stylesDir, 'zju-system-design.sty'), style);

const main = String.raw`%!TeX program = xelatex
\documentclass[12pt,a4paper,UTF8,openany]{ctexrep}
\usepackage{styles/zju-system-design}
\begin{document}
\pagenumbering{gobble}
\ZJUCover
\clearpage
\pagenumbering{Roman}
\tableofcontents
\clearpage
\pagenumbering{arabic}
\input{chapters/01_overview}
\input{chapters/02_requirements}
\input{chapters/03_architecture}
\input{chapters/04_pose_features}
\input{chapters/05_protogcn}
\input{chapters/06_llm_teacher}
\input{chapters/07_collaboration}
\input{chapters/08_distillation}
\input{chapters/09_incremental_learning}
\input{chapters/10_web_visualization}
\input{chapters/11_code_structure}
\input{chapters/12_datasets}
% 第 13 章“运行环境与部署设备”按当前版本要求暂不纳入文档。
\setcounter{chapter}{13}
\input{chapters/14_completion}
\input{chapters/15_references}
\end{document}
`;
await fs.writeFile(path.join(root, 'main.tex'), main);
if (!await fs.stat(path.join(root, 'bibliography.bib')).then(() => true).catch(() => false)) {
  await fs.writeFile(path.join(root, 'bibliography.bib'), '% Add IEEE BibTeX entries here.\n');
}
await fs.writeFile(path.join(root, 'README.md'), '# 系统设计文档\n\n使用 XeLaTeX 编译：`xelatex main.tex`（连续运行两次以生成目录）。\n\n`source/notion-export.json` 是从 Notion 导出的原始内容；`tools/generate_latex.mjs` 将其排版为本工程。\n');

async function download(url, output) {
  try { await fs.access(output); return; } catch { /* download missing file */ }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  const response = await fetch(url, { signal: controller.signal });
  clearTimeout(timeout);
  if (!response.ok) throw new Error(`Download failed (${response.status}): ${url}`);
  await fs.writeFile(output, Buffer.from(await response.arrayBuffer()));
}
for (const [id, filename] of imageNames) {
  const block = findBlock(source.blocks, id);
  const image = block.image;
  await download(image.type === 'file' ? image.file.url : image.external.url, path.join(figuresDir, filename));
}
async function downloadZjuLogo(output) {
  try { await fs.access(output); return; } catch { /* download missing file */ }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch('https://api.github.com/repos/haochengxia/zjureport/contents/figures/zju_logo.png', {
      signal: controller.signal,
      headers: { Accept: 'application/vnd.github+json' },
    });
    if (!response.ok) throw new Error(`Download failed (${response.status}).`);
    const asset = await response.json();
    await fs.writeFile(output, Buffer.from(asset.content.replace(/\n/g, ''), 'base64'));
  } finally {
    clearTimeout(timeout);
  }
}
try { await downloadZjuLogo(path.join(figuresDir, 'zju_logo.png')); }
catch (error) { console.warn(`Zhejiang University logo download skipped: ${error.message}`); }

function findBlock(blocks, id) {
  for (const block of blocks) {
    if (block.id === id) return block;
    const child = findBlock(block.children ?? [], id);
    if (child) return child;
  }
  return null;
}

console.log(`Generated ${chapterFiles.length} chapters and downloaded ${imageNames.size} Notion images.`);
