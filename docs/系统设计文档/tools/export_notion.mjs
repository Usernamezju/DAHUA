import fs from 'node:fs/promises';
import https from 'node:https';

const pageId = '3ca44bfd-bea5-8004-a2ce-f05b25794fe6';
const config = await fs.readFile('/home/fjp/.codex/config.toml', 'utf8');
const token = config.match(/NOTION_TOKEN\s*=\s*"([^"]+)"/)?.[1];
if (!token) throw new Error('NOTION_TOKEN not found in Codex configuration.');

function request(path) {
  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: 'api.notion.com', path, method: 'GET',
      headers: { Authorization: `Bearer ${token}`, 'Notion-Version': '2022-06-28' },
    }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        if (res.statusCode < 200 || res.statusCode >= 300) return reject(new Error(`${res.statusCode}: ${body}`));
        resolve(JSON.parse(body));
      });
    });
    req.on('error', reject);
    req.end();
  });
}

async function children(blockId) {
  const result = [];
  let cursor = '';
  do {
    const page = await request(`/v1/blocks/${blockId}/children?page_size=100${cursor ? `&start_cursor=${cursor}` : ''}`);
    result.push(...page.results);
    cursor = page.has_more ? page.next_cursor : '';
  } while (cursor);
  return result;
}

async function expand(block) {
  if (block.has_children) block.children = await Promise.all((await children(block.id)).map(expand));
  return block;
}

const page = await request(`/v1/pages/${pageId}`);
const blocks = await Promise.all((await children(pageId)).map(expand));
await fs.mkdir(new URL('../source/', import.meta.url), { recursive: true });
await fs.writeFile(new URL('../source/notion-export.json', import.meta.url), JSON.stringify({ page, blocks }, null, 2));
console.log(`Exported ${blocks.length} top-level blocks to source/notion-export.json`);
