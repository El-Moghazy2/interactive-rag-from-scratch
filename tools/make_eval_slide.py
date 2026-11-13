"""Rebuild the evaluation 'bars' slide in both decks: three rows (retrieval, generation, end to end),
each with its score bars on the left and a small picture of what that score looks like on the right.

Usage: python tools/make_eval_slide.py   (safe to run again)
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SLIDE = '''<section class="slide" id="ev-bars" data-part="9" style="gap:11px">
  <div class="kicker"><b>9</b> · Evaluation · play</div>
  <h2>Drag a score. See what it means, and what to fix first.</h2>
  <div class="chips" id="eb-presets"></div>
  <div class="col grow" id="eb-rows" style="gap:9px;min-height:0"></div>
  <div class="card" id="eb-head" style="font-size:18px;padding:11px 18px;transition:all .3s"></div>
  <p class="small" style="font-size:14px">You set the levels; nothing here is measured. The pictures are sketches: four retrieved chunks, a four-sentence answer, four facts in the reference answer.</p>
</section>'''

DEMO = r'''  /* ================= Part 9: evaluation bars ================= */
  DEMOS['ev-bars'] = {
    init(s) {
      const M = {
        recall: { name: 'Context recall', fix: 'split the text into better chunks, retrieve more of them, or add keyword search' },
        precision: { name: 'Context precision', fix: 'add a reranker, or retrieve fewer chunks' },
        faith: { name: 'Faithfulness', fix: 'tighten the prompt (answer only from the context) and lower the temperature' },
        rel: { name: 'Response relevancy', fix: 'ask for short, direct answers in the prompt' },
        acc: { name: 'Answer accuracy', fix: 'read ten wrong answers beside their chunks, and check the reference answers' },
      };
      const GROUPS = [
        { title: 'Retrieval', q: 'did the right text reach the model?', ids: ['recall', 'precision'], pic: 'ret' },
        { title: 'Generation', q: 'did the model use that text well?', ids: ['faith', 'rel'], pic: 'gen' },
        { title: 'End to end', q: 'is the final answer right?', ids: ['acc'], pic: 'acc' },
      ];
      const IDS = Object.keys(M);
      const PRESETS = [
        ['Healthy', [.9, .9, .95, .85, .9]],
        ['Retrieval misses', [.35, .7, .85, .8, .4]],
        ['Buried in noise', [.9, .3, .7, .65, .7]],
        ['Making things up', [.9, .85, .3, .8, .45]],
        ['Faithful but wrong', [.55, .5, .95, .85, .2]],
        ['Dodging the question', [.85, .8, .6, .25, .3]],
      ];
      const band = v => v >= 0.8 ? 2 : v >= 0.5 ? 1 : 0;
      const col = v => ['var(--bad)', 'var(--work)', 'var(--hit)'][band(v)];
      const rows = $('#eb-rows', s);
      rows.innerHTML = GROUPS.map(g => `<div class="ev2 panel">
        <div class="ev2-bars"><h3>${g.title} <span>${g.q}</span></h3>
          ${g.ids.map(id => `<div class="ev2-row"><b>${M[id].name}</b><input type="range" class="ebar" id="eb-${id}" min="0" max="1" step="0.05" value="0.9" aria-label="${M[id].name}"><output id="eb-${id}-o"></output></div>`).join('')}
        </div>
        <div class="ev2-pic" id="ep-${g.pic}"></div></div>`).join('');

      const pics = {
        ret(v) {                               // four retrieved chunks, two of which the answer needs
          const found = v.recall >= 0.75 ? 2 : v.recall >= 0.25 ? 1 : 0, noise = 4 - found;
          const u = Array(found).fill('u'), n = Array(noise).fill('n');
          const order = !found || v.precision >= 0.8 ? [...u, ...n] : v.precision >= 0.5 ? ['n', ...u, ...n.slice(1)] : [...n, ...u];
          const blocks = order.map((k, i) => `<span class="ck ${k}"><i>#${i + 1}</i>${k === 'u' ? 'useful' : 'noise'}</span>`).join('') +
            Array(2 - found).fill('<span class="ck miss">needed,<br>not retrieved</span>').join('');
          const cap = (found === 2 ? 'Both chunks the answer needs were retrieved' : found === 1 ? 'One chunk the answer needs is missing' : 'Neither chunk the answer needs was retrieved') +
            (!found ? '.' : v.precision >= 0.8 ? ', and the useful ones are on top.' : v.precision >= 0.5 ? ', with some noise ranked above.' : ', but the useful ones are buried under noise.');
          return `<div class="picrow"><em>retrieved, in rank order</em>${blocks}</div><p>${cap}</p>`;
        },
        gen(v) {                               // a four-sentence answer
          const flagged = x => x >= 0.8 ? 0 : Math.max(1, Math.round((1 - x) * 4));   // a healthy score flags nothing
          const bad = flagged(v.faith), off = flagged(v.rel), w = [92, 78, 86, 64];
          const lines = w.map((x, i) => {
            const uns = i >= 4 - bad, offt = i >= 4 - off;
            return `<div class="aline${uns ? ' uns' : ''}${offt ? ' off' : ''}"><i style="width:${x * 0.5}%"></i><b>${uns ? 'not in any chunk' : 'backed by a chunk'}</b>${offt ? '<u>off the question</u>' : ''}</div>`;
          }).join('');
          const cap = (bad === 0 ? 'Every sentence is backed by a chunk' : bad === 4 ? 'No sentence is backed by a chunk' : `${bad} of 4 sentences ${bad > 1 ? 'are' : 'is'} not backed by any chunk`) +
            (off === 0 ? ', and all of it answers the question.' : off >= 3 ? ', and most of it does not answer the question.' : `, and ${off} ${off > 1 ? 'drift' : 'drifts'} away from the question.`);
          return `<div class="alines">${lines}</div><p>${cap}</p>`;
        },
        acc(v) {                               // four facts in the reference answer
          const m = Math.round(v.acc * 4);
          const facts = [0, 1, 2, 3].map(i => `<span class="fact ${i < m ? 'y' : 'n'}">${i < m ? '✓' : '✗'}</span>`).join('');
          return `<div class="picrow"><em>facts in the reference answer</em>${facts}<p style="margin:0 0 0 10px">${m} of 4 match. ${['The answer disagrees with the reference.', 'The answer is only partly right.', 'The answer agrees with the reference.'][band(v.acc)]}</p></div>`;
        },
      };

      const draw = () => {
        const v = Object.fromEntries(IDS.map(id => [id, +$('#eb-' + id, s).value]));
        IDS.forEach(id => {
          const el = $('#eb-' + id, s);
          el.style.setProperty('--fill', col(v[id])); el.style.setProperty('--pct', v[id] * 100 + '%');
          const o = $(`#eb-${id}-o`, s); o.textContent = v[id].toFixed(2); o.style.color = col(v[id]);
        });
        GROUPS.forEach(g => { $('#ep-' + g.pic, s).innerHTML = pics[g.pic](v); });
        const low = id => v[id] < 0.5, ok = id => v[id] >= 0.8;
        let head, tone = 'bad';
        if (IDS.every(ok)) { head = '<b>Healthy.</b> Keep these numbers as your baseline and run the same questions again after every change.'; tone = 'hit'; }
        else if (low('recall')) head = '<b>Start with retrieval.</b> The text the answer needs is not reaching the model, and no prompt can fix that.';
        else if (low('acc') && ok('faith')) head = '<b>Faithful but wrong.</b> The model repeats its context accurately, and the context is wrong or incomplete. Look at retrieval before the prompt.';
        else if (low('faith')) head = '<b>A generation problem.</b> Retrieval is delivering, and the model is not sticking to what it was given.';
        else if (low('precision')) head = '<b>Found, but buried.</b> The right chunks are retrieved and outranked by noise.';
        else if (low('rel')) head = '<b>Not answering the question.</b> The answers may be true and supported, but they do not address what was asked.';
        else if (low('acc')) head = '<b>Upstream looks fine, yet answers disagree with the references.</b> Read some cases by hand.';
        else { head = '<b>No single failure.</b> Nothing is broken, but there is room to improve.'; tone = 'work'; }
        const todo = IDS.filter(id => !ok(id)).sort((a, b) => v[a] - v[b]).slice(0, 2);
        const steps = todo.map((id, i) => `<span class="step"><em>${i ? 'Then' : 'First'}</em> <b style="color:${col(v[id])}">${M[id].name} ${v[id].toFixed(2)}</b>: ${M[id].fix}.</span>`).join('');
        const h = $('#eb-head', s); h.className = 'card ' + tone;
        h.innerHTML = head + (steps ? `<div class="steps">${steps}</div>` : '');
        $('#eb-presets', s).querySelectorAll('.chip').forEach((c, i) => c.classList.toggle('sel', PRESETS[i][1].every((x, k) => x === v[IDS[k]])));
      };
      PRESETS.forEach(([label, vals]) => {
        const c = document.createElement('button');
        c.className = 'chip'; c.textContent = label;
        c.onclick = () => { IDS.forEach((id, k) => $('#eb-' + id, s).value = vals[k]); draw(); };
        $('#eb-presets', s).appendChild(c);
      });
      rows.querySelectorAll('input').forEach(e => e.oninput = draw);
      draw();
    },
  };
'''

CSS = """
/* evaluation slide, version 2 */
.ev2 { display: grid; grid-template-columns: 1fr 1.05fr; gap: 26px; align-items: center; padding: 9px 18px; }
.ev2-bars h3 { font-size: 15px; letter-spacing: .1em; text-transform: uppercase; color: var(--ink); margin: 0 0 4px; }
.ev2-bars h3 span { text-transform: none; letter-spacing: 0; font-weight: 400; color: var(--muted); font-size: 15.5px; margin-left: 6px; }
.ev2-row { display: grid; grid-template-columns: 178px 1fr 54px; gap: 12px; align-items: center; padding: 4px 0; font-size: 17px; }
.ev2-row output { font-family: var(--mono); font-size: 22px; font-weight: 800; text-align: right; transition: color .2s; }
.ev2-pic { font-size: 14.5px; color: var(--muted); min-width: 0; }
.ev2-pic p { margin: 5px 0 0; color: var(--ink); font-size: 15px; }
.picrow { display: flex; align-items: center; gap: 6px; flex-wrap: nowrap; }
.picrow em { font-style: normal; font-size: 12.5px; width: 86px; line-height: 1.2; flex: none; }
.ck { display: inline-flex; flex-direction: column; align-items: center; justify-content: center; width: 66px; height: 44px; border-radius: 8px; font-size: 13px; font-weight: 700; line-height: 1.15; text-align: center; transition: all .25s; }
.ck i { font-style: normal; font-family: var(--mono); font-size: 11px; font-weight: 500; opacity: .75; }
.ck.u { background: var(--hit); color: #fff; } .ck.n { background: var(--line); color: var(--muted); }
.ck.miss { width: 88px; background: transparent; border: 2px dashed var(--bad); color: var(--bad); font-size: 11.5px; font-weight: 600; }
.alines { display: flex; flex-direction: column; gap: 4px; }
.aline { display: flex; align-items: center; gap: 8px; font-size: 12.5px; }
.aline i { height: 9px; border-radius: 5px; background: var(--hit); flex: none; transition: all .25s; }
.aline b { font-weight: 600; color: var(--hit); } .aline.uns i { background: var(--bad); } .aline.uns b { color: var(--bad); }
.aline u { text-decoration: none; border: 1.5px solid var(--work); color: var(--work); border-radius: 99px; padding: 0 7px; font-weight: 600; }
.aline.off i { opacity: .45; }
.fact { display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 8px; font-weight: 800; font-size: 17px; flex: none; }
.fact.y { background: var(--hit); color: #fff; } .fact.n { border: 2px dashed var(--bad); color: var(--bad); }
#eb-head .steps { display: flex; gap: 22px; margin-top: 6px; font-size: 15.5px; }
#eb-head .step { flex: 1; } #eb-head .step em { font-style: normal; font-weight: 800; text-transform: uppercase; font-size: 12px; letter-spacing: .1em; color: var(--muted); margin-right: 4px; }
"""

for deck in ["site", "site-short"]:
    p = ROOT / deck / "index.html"
    s = p.read_text(encoding="utf-8")
    a = s.index('<section class="slide" id="ev-bars"')
    b = s.index("</section>", a) + len("</section>")
    s = s[:a] + SLIDE + s[b:]
    s = re.sub(r'\?v=(\d+)"', lambda m: f'?v={int(m.group(1)) + 1}"', s)
    p.write_text(s, encoding="utf-8")

    j = ROOT / deck / "demos.js"
    t = j.read_text(encoding="utf-8")
    a = t.index("  /* ================= Part 9: evaluation bars")
    b = t.rindex("})();")
    j.write_text(t[:a] + DEMO + t[b:], encoding="utf-8")

    c = ROOT / deck / "style.css"
    css = c.read_text(encoding="utf-8")
    if "/* evaluation slide, version 2 */" in css:
        x = css.index("\n/* evaluation slide, version 2 */")
        y = css.find("\n/* ", x + 5)
        css = css[:x] + (css[y:] if y > 0 else "")
    c.write_text(css + CSS, encoding="utf-8")
print("evaluation slide rebuilt in both decks")
