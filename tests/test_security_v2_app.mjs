import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { runInNewContext } from 'node:vm';

class Element {
  constructor() {
    this.children = [];
    this.events = {};
    this.hidden = false;
    this.style = {};
    this.textContent = '';
    this.value = '';
    this.classList = { toggle() {} };
  }
  addEventListener(name, callback) { this.events[name] = callback; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = [...children]; }
  setAttribute() {}
  focus() {}
}

function page(prediction) {
  const html = readFileSync(new URL('../security_jev_v2/app.html', import.meta.url), 'utf8');
  const script = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const document = { getElementById: get, createElement: () => new Element(), addEventListener() {} };
  const calls = [];
  const fetch = async url => {
    calls.push(url);
    return { ok: true, json: async () => url.endsWith('/info')
      ? {model_version: 'test-v2', device: 'cpu'} : prediction };
  };
  runInNewContext(script, { document, fetch, console });
  return { get, calls };
}

const prediction = {
  model_version: 'test-v2', elapsed_ms: 10.1, truncated: false, retained_ratio: 1,
  noul: { probabilities: { '是': 0.531234, '否': 0.468766 }, selected: '是' },
  choice: { probabilities: { 'SQL注入': .251234, 'XSS': .251234,
    '路径遍历': .251234, '其他攻击': .246298 }, selected: 'SQL注入', applicable: true },
  policy: { experimental_threshold: .9, would_block: false }
};

test('v2 page uses v2 API and keeps model answer separate from policy', async () => {
  const { get, calls } = page(prediction);
  get('request-input').value = 'GET / HTTP/1.1';
  await get('analyze-button').events.click();
  assert.ok(calls.includes('/api/v2/predict'));
  assert.match(get('verdict').textContent, /放行/);
  const binary = get('block-options').children;
  assert.equal(binary.filter(option => option.className.includes('selected')).length, 1);
  assert.equal(binary.reduce((sum, option) => sum + Number(option.children[0].children.at(-1).textContent.replace('%', '')), 0), 100);
  const types = get('type-options').children;
  assert.equal(types.reduce((sum, option) => sum + Number(option.children[0].children.at(-1).textContent.replace('%', '')), 0), 100);
});

test('normal prediction shows conditional type scores with no selected type', async () => {
  const { get } = page({ ...prediction,
    noul: { probabilities: { '是': .2, '否': .8 }, selected: '否' },
    choice: { ...prediction.choice, applicable: false, selected: null } });
  get('request-input').value = 'GET / HTTP/1.1';
  await get('analyze-button').events.click();
  assert.equal(get('type-options').children.filter(option => option.className.includes('selected')).length, 0);
  assert.match(get('choice-caption').textContent, /不适用/);
});
