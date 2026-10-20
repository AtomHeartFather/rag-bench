#!/usr/bin/env python3
"""RAG-benchmark stand harness.

Subcommands:
  gen-corpus  --scale 1k [--limit N]   # world + docs + chunks via local LLM
  gen-queries                    # 300 queries (5 levels x 60), scale-agnostic
  index       --variant file|fts|rag --scale 1k
  search      --variant file|fts|rag --scale 1k [--k 10] [--limit N]
  judge       --variant rag|fts --scale 1k [--sample 60]
  run-all     --variant V --scale S [--judge]

All outputs under BASE/{corpus,queries,results,logs}.
"""
import argparse, json, os, re, sys, time, random, hashlib, resource, shutil, threading, subprocess, statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent.parent
QDRANT = "http://127.0.0.1:6333"
EMBED = "http://127.0.0.1:8002/v1"
LLM = "http://127.0.0.1:13305/api/v1"
LLM_MODEL = os.environ.get("BENCH_LLM_MODEL", "local-llm")
EMBED_MODEL_CANDIDATES = ["bge-m3", "bge-m3-Q4_K_M.gguf"]
EMBED_DIM = 1024
CHUNK_TARGET = 1500
SCALES = {"1k": 100, "10k": 1000, "50k": 5000, "100k": 10000,
          # real-data scales (corpus/raw_real): sizes are placeholder; see corpus/chunks_*.jsonl
          "real-rag-10k": 10000, "real-rag-66k": 66581, "real-mh": 609}
ENTITY_MIX = {"1k": (50, 25, 25)}  # clients, vendors, projects for 1k
HOST_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "HOST"]  # host for RAM sampling of Qdrant/embed server

TYPE_RU = {"client": "клиент", "vendor": "вендор", "project": "проект"}
TYPE_PREFIX = {"client": "c", "vendor": "v", "project": "p"}
DOC_TYPES = ["profile", "contract_sla", "incidents", "payment_policy", "chat"]
DOC_RU = {"profile": "профиль", "contract_sla": "договор и SLA",
          "incidents": "журнал инцидентов", "payment_policy": "политика оплаты", "chat": "чат-транскрипт"}

# 42 fact fields per entity: (field, doc_type, ru_label, kind)
FIELDS = [
    ("industry", "profile", "отрасль", "str"), ("founded_year", "profile", "год основания", "int"),
    ("city", "profile", "город", "str"), ("website", "profile", "сайт", "str"),
    ("phone", "profile", "телефон", "str"), ("email", "profile", "e-mail", "str"),
    ("contact_name", "profile", "контактное лицо", "str"),
    ("employee_count", "profile", "численность сотрудников", "int"), ("status", "profile", "статус", "str"),
    ("contract_id", "contract_sla", "номер договора", "str"), ("start_date", "contract_sla", "дата начала договора", "date"),
    ("end_date", "contract_sla", "дата окончания договора", "date"), ("amount_usd", "contract_sla", "стоимость договора", "num"),
    ("payment_cycle", "contract_sla", "цикл оплаты", "str"),
    ("sla_uptime_pct", "contract_sla", "SLA аптайм", "num"), ("response_time_h", "contract_sla", "время реакции SLA", "num"),
    ("penalty_pct", "contract_sla", "штраф за нарушение SLA", "num"), ("auto_renew", "contract_sla", "авторасширение договора", "yesno"),
]
for i in range(1, 7):
    FIELDS.append((f"incident{i}_date", "incidents", f"дата инцидента {i}", "date"))
    FIELDS.append((f"incident{i}_severity", "incidents", f"серьёзность инцидента {i}", "str"))
    FIELDS.append((f"incident{i}_duration_h", "incidents", f"продолжительность инцидента {i}", "num"))
FIELDS += [("last_payment_date", "payment_policy", "дата последнего платежа", "date"),
           ("last_payment_amount_usd", "payment_policy", "сумма последнего платежа", "num"),
           ("dpo_days", "payment_policy", "DPO (срок оплаты)", "int"),
           ("late_fee_pct", "payment_policy", "процент за просрочку", "num"),
           ("last_chat_date", "chat", "дата последнего чата", "date"),
           ("pending_issue", "chat", "открытый вопрос", "str")]
assert len(FIELDS) == 42, len(FIELDS)
FIELD_BY_NAME = {f[0]: f for f in FIELDS}

ROOTS = ['Омега', 'Вектор', 'Полюс', 'Гранит', 'Меридиан', 'Северный', 'Атлас', 'Квадрат', 'Рубин',
         'Спектр', 'Транзит', 'Фосфор', 'Электрон', 'Заря', 'Импульс', 'Комета', 'Лазурь', 'Мираж',
         'Орион', 'Прайм', 'Радуга', 'Сфинкс', 'Терра', 'Уран', 'Феникс', 'Зенит', 'Изумруд',
         'Кайрос', 'Левша', 'Метеор', 'Навигатор', 'Оникс', 'Палладий', 'Радиант', 'Синтез',
         'Тайга', 'Ультра', 'Циклон', 'Шторм', 'Арктика', 'Бриз', 'Гелиос', 'Драйв', 'Единство',
         'Контур', 'Логистика', 'Маяк', 'Нейрон', 'Октава', 'Панорама', 'Прометей', 'Рубеж',
         'Соларис', 'Универ', 'Формула', 'Якорь', 'Кобальт', 'Люмен']
CITIES = ['Москва', 'Санкт-Петербург', 'Казань', 'Новосибирск', 'Екатеринбург', 'Нижний Новгород',
          'Краснодар', 'Самара', 'Ростов-на-Дону', 'Уфа']
SUFFIXES = {'client': ['Тех', 'Софт', 'Групп', 'Девайсы', 'Лаб', 'Сервис'],
            'vendor': ['Пром', 'Снаб', 'Комп', 'Торг', 'Логистик', 'Парт'],
            'project': ['Альфа', 'Бета', 'Гамма', 'Дельта', 'Эпсилон', 'Омега']}

STOP = set("и в во не на у о а со к за от но по при до для с что как где".split())
EN_STOP = set("the a an and or but if then else of to in on at by for with from as is are was were be been being "
              "do does did have has had this that these those it its i you he she we they them his her their "
              "my our your who whom whose what which when where why how not no yes can could should would will may "
              "might must shall about into over under again further once here there all any both each few more most "
              "other some such only own same so than too very s t don now".split())


def tokenize(t):
    t = t.lower()
    return [x for x in re.findall(r'[a-zа-яё0-9]+', t) if len(x) > 1 and x not in STOP]


# naive RU suffix stem (applied identically on index and query side)
RU_SUFFIXES = ['ия', 'ой', 'ая', 'ую', 'ого', 'его', 'ими', 'ами', 'ыми', 'ов', 'ев', 'ий', 'ая',
               'ое', 'ом', 'ем', 'у', 'а', 'е', 'я', 'о', 'и', 'ы', 'ь']
def stem_lite(t):
    for s in RU_SUFFIXES:
        if t.endswith(s) and len(t) - len(s) >= 2:
            return t[:len(t) - len(s)]
    return t


def stem_tokens(toks):
    return [stem_lite(x) for x in toks]


def sha_id(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:15], 16)


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


# ---------------- LLM (local, OpenAI-compatible) ----------------
class LLMStats:
    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
    def add(self, usage):
        self.calls += 1
        if usage:
            self.prompt_tokens += usage.get('prompt_tokens', 0)
            self.completion_tokens += usage.get('completion_tokens', 0)


def llm_chat(messages, temperature=0.2, max_tokens=1000, stats=None, retries=3,
             system=None, thinking=False):
    if system:
        messages = [{"role": "system", "content": system}] + messages
    for a in range(retries):
        try:
            r = requests.post(f"{LLM}/chat/completions",
                              json={"model": LLM_MODEL, "messages": messages,
                                    "temperature": temperature, "max_tokens": max_tokens,
                                    "enable_thinking": thinking},
                              timeout=900)
            r.raise_for_status()
            d = r.json()
            if stats:
                stats.add(d.get('usage'))
            return d['choices'][0]['message'].get('content') or ''
        except Exception as e:
            if a == retries - 1:
                raise
            time.sleep(2 * (a + 1))


def extract_json(text):
    t = re.sub(r'^```(?:json)?\s*', '', text.strip()).rsplit('```', 1)[0].strip()
    i, j = t.find('['), t.find('{')
    starts = [x for x in (i, j) if x >= 0]
    if not starts:
        raise ValueError(f'no json in: {text[:200]}')
    s = min(starts)
    open_c, close_c = ('{', '}') if t[s] == '{' else ('[', ']')
    depth = 0
    for k in range(s, len(t)):
        if t[k] == open_c: depth += 1
        elif t[k] == close_c:
            depth -= 1
            if depth == 0:
                return json.loads(t[s:k + 1])
    raise ValueError('unbalanced json')


# ---------------- world skeleton ----------------
def gen_world(n_entities, rng):
    n_cl, n_vd, n_pr = ENTITY_MIX["1k"] if n_entities == 100 else (
        int(0.5 * n_entities), int(0.25 * n_entities), n_entities - int(0.5 * n_entities) - int(0.25 * n_entities))
    ents = []
    for etype, n, base in (("client", n_cl, 0), ("vendor", n_vd, 200), ("project", n_pr, 400)):
        for i in range(n):
            cluster = i // 6
            suffix_i = i % 6
            root = ROOTS[(cluster * 7 + base) % len(ROOTS)]
            suffix = SUFFIXES[etype][suffix_i]
            if etype == 'project':
                name = f"Проект «{root}-{suffix}»"
            else:
                name = f"«{root}-{suffix}»"
            eid = f"{TYPE_PREFIX[etype]}{i + 1:03d}"
            ents.append({"id": eid, "type": etype, "name": name,
                         "city": CITIES[rng.randrange(len(CITIES))], "cluster": f"{etype}-{cluster}"})
    return ents[:n_entities]


# ---------------- corpus gen ----------------
def llm_facts_prompt(e):
    keys = ", ".join(f[0] for f in FIELDS)
    return f"""Сгенерируй реалистичные данные для сущности из синтетического мира CRM.
Тип: {TYPE_RU[e['type']]}, имя: {e['name']}, город: {e['city']}.
Верни строго JSON-объект (без комментариев) с ровно 42 ключами: {keys}.
Требования:
- даты в формате YYYY-MM-DD, период 2023-01-01 по 2026-09-30; start_date < end_date; last_payment_date < 2026-10-01;
- contract_id вида KTR-YYYY-NNNN; incidentN_date неубывающие;
- severity из ["low","medium","high","critical"]; status из ["active","paused","churned"];
- payment_cycle из ["monthly","quarterly","yearly"]; auto_renew: "да" или "нет"; валюта везде USD;
- amount_usd, last_payment_amount_usd: целые числа; sla_uptime_pct: 99.0-99.99; response_time_h: 1-24;
- employee_count: 5-500; dpo_days: 15-60; late_fee_pct, penalty_pct: 0.1-5;
- имена: русские ФИО; website: короткое латинское доменное имя; phone: +7 9XX XXX-XX-XX;
- pending_issue: короткая фраза 5-10 слов об открытой технической проблеме.
Только JSON."""


def render_docs(e, f):
    """Render 5 documents as markdown from the 42 fact values."""
    def g(field):
        return f[field]
    d = {}
    d['profile'] = f"""# Профиль: {e['name']}

Тип сущности: {TYPE_RU[e['type']]}, ID: {e['id']}

## Контактная информация
- Телефон: {g('phone')}
- E-mail: {g('email')}
- Сайт: {g('website')}
- Контактное лицо: {g('contact_name')}

## Общие сведения
- Город: {g('city')}
- Отрасль: {g('industry')}
- Год основания: {g('founded_year')}
- Численность сотрудников: {g('employee_count')}
- Статус: {g('status')}
"""
    d['contract_sla'] = f"""# Договор и SLA: {e['name']}

ID сущности: {e['id']}

## Условия договора
- Номер договора: {g('contract_id')}
- Дата начала: {g('start_date')}
- Дата окончания: {g('end_date')}
- Стоимость договора: {g('amount_usd')} USD
- Валюта: USD
- Цикл оплаты: {g('payment_cycle')}
- Авторасширение: {g('auto_renew')}

## SLA
- Гарантированный аптайм: {g('sla_uptime_pct')}%
- Время реакции на инцидент: {g('response_time_h')} ч
- Штраф за нарушение SLA: {g('penalty_pct')}% от месячной оплаты
"""
    rows = "\n".join(
        f"| {g(f'incident{i}_date')} | {g(f'incident{i}_severity')} | {g(f'incident{i}_duration_h')} ч |"
        for i in range(1, 7))
    d['incidents'] = f"""# Журнал инцидентов: {e['name']}

ID сущности: {e['id']}. Зафиксировано 6 инцидентов.

| Дата | Серьёзность | Длительность |
|---|---|---|
{rows}
"""
    d['payment_policy'] = f"""# Политика оплаты: {e['name']}

ID сущности: {e['id']}

## Условия
- Валюта расчётов: USD
- DPO (срок оплаты счёта): {g('dpo_days')} дней
- Цикл выставления счетов: {g('payment_cycle')}
- Процент за просрочку платежа: {g('late_fee_pct')}% в месяц

## Последние платежи
- Дата последнего платежа: {g('last_payment_date')}
- Сумма последнего платежа: {g('last_payment_amount_usd')} USD
"""
    d['chat'] = f"""# Чат-транскрипт: {e['name']}

ID сущности: {e['id']}. Последнее сообщение: {g('last_chat_date')}.

{g('contact_name')}: Добрый день! Хотел уточнить статус по договорённости из договора {g('contract_id')}.
Менеджер: Здравствуйте! По {TYPE_RU[e['type']]} {e['name']} договоренности выполняются, цикл оплаты {g('payment_cycle')}.
{g('contact_name')}: А по инциденту {g('incident1_date')}? Клиенты жалуются, это было {g('incident1_severity')}.
Менеджер: Да, тот инцидент длился {g('incident1_duration_h')} ч,SLA-реакция {g('response_time_h')} ч соблюдена.
{g('contact_name')}: Хорошо. И ещё — что с нашим открытым вопросом?
Менеджер: Открытый вопрос: {g('pending_issue')}. Возьмём в работу до конца следующей недели.
"""
    return d


def split_long(b, target):
    parts = []
    while len(b) > target:
        cut = max(b.rfind('. ', 0, target), b.rfind('! ', 0, target),
                  b.rfind('? ', 0, target), b.rfind('\n', 0, target))
        if cut < 60:
            cut = target
        parts.append(b[:cut])
        b = b[cut:].lstrip()
    if b.strip():
        parts.append(b)
    return parts


def make_chunks(doc_text, target=CHUNK_TARGET):
    blocks = re.split(r'\n{2,}', doc_text)
    out, cur = [], ""
    def add(s):
        nonlocal cur
        if not s.strip():
            return
        if cur and len(cur) + len(s) + 2 > target:
            out.append(cur.strip())
            cur = s
        else:
            cur = (cur + "\n\n" + s) if cur else s
    for b in blocks:
        for p in split_long(b, target):
            add(p)
    if cur.strip():
        out.append(cur.strip())
    return [c for c in out if len(c) > 40]


def corpus_paths(scale):
    return {"world": BASE / "corpus" / f"world_{scale}.json",
            "chunks": BASE / "corpus" / f"chunks_{scale}.jsonl",
            "llm_dir": BASE / "corpus" / "llm_cache"}


def queries_path(scale):
    """Per-scale queries file if present (real datasets), else shared queries.jsonl."""
    p = BASE / "queries" / f"queries_{scale}.jsonl"
    return p if p.exists() else BASE / "queries" / "queries.jsonl"


def cmd_gen_corpus(args):
    scale = args.scale
    n = SCALES[scale]
    if args.limit:
        n = min(n, args.limit)
    p = corpus_paths(scale)
    p['llm_dir'].mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    ents = gen_world(n, rng)
    stats = LLMStats()
    t0 = time.time()

    def gen_one(e):
        cache = p['llm_dir'] / f"{e['id']}.json"
        if cache.exists() and not args.force:
            return json.loads(cache.read_text())
        msgs = [{"role": "user", "content": llm_facts_prompt(e)}]
        for a in range(4):
            txt = llm_chat(msgs, temperature=0.2, max_tokens=1500, stats=stats,
                           system="Работай быстро: /no_think, минимальное рассуждение, затем ответ.")
            try:
                data = extract_json(txt)
                got = set(data.keys())
                need = {f[0] for f in FIELDS}
                if need <= got:
                    f = {k: data[k] for k in need}
                    f['status'] = str(f['status'])
                    f['auto_renew'] = str(f['auto_renew'])
                    cache.write_text(json.dumps(f, ensure_ascii=False))
                    return f
            except Exception:
                pass
            msgs = [{"role": "user", "content": llm_facts_prompt(e) +
                     "\n\n(попытка " + str(a + 2) + ": предыдущий ответ не был валидным JSON со всеми 42 ключами — проверь)"}]
        raise RuntimeError(f'facts failed for {e["id"]}')

    facts, failures = {}, []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [(e, ex.submit(gen_one, e)) for e in ents]
        for e, fu in futs:
            try:
                facts[e['id']] = fu.result()
            except Exception as ex2:
                failures.append((e['id'], str(ex2)))
    if failures:
        raise RuntimeError(f'failed entities: {failures}')
    t_llm = time.time() - t0

    # render docs + chunk (shared chunking for all variants)
    chunks, fact_chunk = [], {}
    for e in ents:
        f = facts[e['id']]
        docs = render_docs(e, f)
        for dt in DOC_TYPES:
            for idx, ctext in enumerate(make_chunks(docs[dt])):
                cid = f"{e['id']}_{dt}_{idx}"
                text = f"Сущность: {e['name']} ({e['id']}), документ: {DOC_RU[dt]}.\n{ctext}"
                chunks.append({"chunk_id": cid, "entity_id": e['id'], "doc_type": dt,
                               "text": text, "ts": "2026-10-01",
                               "topic_tags": [e['type'], dt]})
        for field, dt, _lbl, _k in FIELDS:
            val = str(f[field])
            hit = None
            for c in chunks:
                if c['entity_id'] == e['id'] and c['doc_type'] == dt and val in c['text']:
                    hit = c['chunk_id']
                    break
            fact_chunk[f"{e['id']}:{field}"] = hit or f"{e['id']}_{dt}_0"
    write_jsonl(p['chunks'], chunks)
    (BASE / 'corpus' / f'world_{scale}.json').write_text(json.dumps(
        {"entities": ents, "facts": facts, "fact_chunk": fact_chunk}, ensure_ascii=False))

    n_chars = sum(len(c['text']) for c in chunks)
    out = {"scale": scale, "entities": len(ents), "facts": len(ents) * 42,
           "chunks": len(chunks), "chars": n_chars,
           "llm_calls": stats.calls, "llm_prompt_tokens": stats.prompt_tokens,
           "llm_completion_tokens": stats.completion_tokens,
           "llm_wall_s": round(t_llm, 1),
           "total_wall_s": round(time.time() - t0, 1)}
    (BASE / 'results' / f'gen_corpus_{scale}.json').parent.mkdir(parents=True, exist_ok=True)
    (BASE / 'results' / f'gen_corpus_{scale}.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False))


# ---------------- queries ----------------
def q_text(field, name, extra=''):
    lbl = FIELD_BY_NAME[field][2]
    if extra:
        return f"Покажите {lbl} и {extra} для сущности {name}."
    return f"Покажите {lbl} для сущности {name}."


AGG_METRICS = [
    ("sum_amount", "sum", "amount_usd", "Суммарная стоимость договоров"),
    ("max_amount", "max", "amount_usd", "Максимальная стоимость договора (назовите значение и имя)"),
    ("min_amount", "min", "amount_usd", "Минимальная стоимость договора (назовите значение и имя)"),
    ("mean_amount", "mean", "amount_usd", "Средняя стоимость договоров"),
    ("sum_emp", "sum", "employee_count", "Общее число сотрудников"),
    ("max_emp", "max", "employee_count", "Наибольшая численность сотрудников (назовите значение и имя)"),
    ("cnt_critical", "cnt", "incident_severity:critical", "Число инцидентов с серьёзностью critical"),
    ("cnt_high", "cnt", "incident_severity:high", "Число инцидентов с серьёзностью high"),
    ("sum_dur", "sum", "incident_duration_h", "Суммарная длительность всех инцидентов (в часах)"),
    ("mean_sla", "mean", "sla_uptime_pct", "Средний SLA аптайм"),
    ("cnt_autorenew", "cnt", "auto_renew:да", "Число сущностей с авторасширением договора"),
    ("cnt_monthly", "cnt", "payment_cycle:monthly", "Число договоров с ежемесячным циклом оплаты"),
    ("sum_lastpay", "sum", "last_payment_amount_usd", "Сумма всех последних платежей"),
    ("mean_dpo", "mean", "dpo_days", "Средний DPO"),
    ("cnt_old", "cnt", "founded_year:lt2000", "Число сущностей, основанных до 2000 года"),
    ("max_founded", "max", "founded_year", "Самый поздний год основания (назовите значение и имя)"),
    ("sum_penalty", "sum", "penalty_pct", "Сумма всех штрафов за нарушение SLA (процентов)"),
    ("mean_resp", "mean", "response_time_h", "Среднее время реакции SLA"),
    ("cnt_active", "cnt", "status:active", "Число сущностей со статусом active"),
    ("min_end", "min", "end_date", "Ранняя дата окончания договора (назовите дату и имя)"),
]


def agg_value(fvals, spec):
    """fvals: list of (entity, facts). spec: (op, field_or_special, q)."""
    op, field = spec[0], spec[1]
    def base(e, f):
        if ':' in field:
            fld, val = field.split(':', 1)
            v = f[fld]
            if val == 'lt2000':
                return 1 if int(v) < 2000 else 0
            return 1 if str(v) == val else 0
        if field == 'incident_severity:critical':
            pass
        return f[field]
    if field in ('incident_severity:critical', 'incident_severity:high'):
        sev = field.split(':')[1]
        vals = [1 for _e, f in fvals for i in range(1, 7) if str(f[f"incident{i}_severity"]) == sev]
        return sum(vals)
    if field == 'incident_duration_h':
        return sum(float(f[f"incident{i}_duration_h"]) for _e, f in fvals for i in range(1, 7))
    if field == 'auto_renew:да':
        return sum(1 for _e, f in fvals if str(f['auto_renew']) == 'да')
    if field == 'payment_cycle:monthly':
        return sum(1 for _e, f in fvals if str(f['payment_cycle']) == 'monthly')
    if field == 'status:active':
        return sum(1 for _e, f in fvals if str(f['status']) == 'active')
    if field == 'founded_year:lt2000':
        return sum(1 for _e, f in fvals if int(f['founded_year']) < 2000)
    if op in ('max', 'min') and not _is_num(field):
        vals = [str(f[field]) for _e, f in fvals]
        return max(vals) if op == 'max' else min(vals)
    if op == 'cnt':
        return None  # handled above
    nums = [float(f[field]) for _e, f in fvals]
    return {"sum": sum(nums), "max": max(nums), "min": min(nums),
            "mean": round(sum(nums) / len(nums), 2)}[op]


def _is_num(field):
    for f in FIELDS:
        if f[0] == field:
            return f[3] in ('num', 'int')
    return False


def cmd_gen_queries(args):
    scale = args.scale
    p = corpus_paths(scale)
    world = json.loads(p['world'].read_text())
    ents, facts, fact_chunk = world['entities'], world['facts'], world['fact_chunk']
    rng = random.Random(42)
    by_type = {}
    for e in ents:
        by_type.setdefault(e['type'], []).append(e)
    # Canonical 1k entity set: stable across scales, so L1-L4 queries are
    # byte-identical for every scale (spec: queries.jsonl common). L5 stays
    # full-scale by design (aggregation over the whole type).
    canon = gen_world(100, random.Random(42))
    by_cluster = {}
    for e in canon:
        by_cluster.setdefault(e['cluster'], []).append(e)

    out = []
    # L1: exact fact, 60
    l1 = []
    pairs = []
    for e in canon:
        for field, dt, lbl, k in FIELDS:
            if k in ('str', 'num', 'int', 'date', 'yesno'):
                pairs.append((e, field))
    rng.shuffle(pairs)
    l1_pairs = pairs[:60]
    for i, (e, field) in enumerate(l1_pairs):
        f = facts[e['id']]
        out.append({"query_id": f"L1-{i + 1:03d}", "level": "L1", "text": q_text(field, e['name']),
                    "entity_ids": [e['id']],
                    "gold_chunk_ids": [fact_chunk[f"{e['id']}:{field}"]],
                    "answer_keys": [str(f[field])], "difficulty": 1, "field": field})
    # L2: paraphrase of L1 via LLM (8 per call), cached
    cache_dir = BASE / 'queries' / 'llm_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    para = {}
    stats = LLMStats()
    l1_list = out[:60]
    for b in range(0, 60, 8):
        batch = l1_list[b:b + 8]
        cf = cache_dir / f"para_{b}.json"
        if cf.exists() and not args.force:
            para.update(json.loads(cf.read_text()))
            continue
        pairs_in = json.dumps([{"id": x['query_id'], "q": x['text']} for x in batch], ensure_ascii=False)
        txt = llm_chat([{"role": "user", "content":
            f"Перефразируйте каждый запрос на естественный русский вопрос (сохранив сущность и поле). "
            f"Вход: {pairs_in}\nВерните JSON-массив объектов {{\"id\":..., \"paraphrase\":...}} точно для всех."}],
            temperature=0.4, max_tokens=800, stats=stats,
            system="Работай быстро: минимальное рассуждение, затем ответ.")
        data = extract_json(txt)
        d = {x['id']: x['paraphrase'] for x in data}
        cf.write_text(json.dumps(d, ensure_ascii=False))
        para.update(d)
    l2_list = []
    for x in out[:60]:
        x2 = dict(x)
        x2['text'] = para.get(x2['query_id'], x['text'])
        x2['level'] = 'L2'
        x2['query_id'] = x2['query_id'].replace('L1', 'L2')
        l2_list.append(x2)
    out = out + l2_list
    # L3: two-chunk (2 fields, different doc_types), 60
    l3cand = []
    for e in canon:
        for f1, d1, _, _ in FIELDS:
            for f2, d2, _, _ in FIELDS:
                if d1 != d2:
                    l3cand.append((e, f1, f2))
    rng.shuffle(l3cand)
    seen = set()
    i = 0
    for e, f1, f2 in l3cand:
        c1, c2 = fact_chunk[f"{e['id']}:{f1}"], fact_chunk[f"{e['id']}:{f2}"]
        if c1 == c2:
            continue
        if (e['id'], f1) in seen and (e['id'], f2) in seen:
            continue
        seen.add((e['id'], f1)); seen.add((e['id'], f2))
        f = facts[e['id']]
        out.append({"query_id": f"L3-{i + 1:03d}", "level": "L3",
                    "text": q_text(f1, e['name'], extra=FIELD_BY_NAME[f2][2]),
                    "entity_ids": [e['id']], "gold_chunk_ids": [c1, c2],
                    "answer_keys": [str(f[f1]), str(f[f2])], "difficulty": 3,
                    "fields": [f1, f2]})
        i += 1
        if i == 60:
            break
    # L4: distractors — entity inside a 6-similar-name cluster, 60
    l4cand = [e for e in ents if len(by_cluster.get(e['cluster'], [])) >= 6]
    rng.shuffle(l4cand)
    l4fields = [f for f, dt, lbl, k in FIELDS if k in ('str', 'num', 'int', 'date')]
    i = 0
    for e in l4cand:
        if i >= 60:
            break
        field = rng.choice(l4fields)
        f = facts[e['id']]
        out.append({"query_id": f"L4-{i + 1:03d}", "level": "L4", "text": q_text(field, e['name']),
                    "entity_ids": [e['id']],
                    "gold_chunk_ids": [fact_chunk[f"{e['id']}:{field}"]],
                    "answer_keys": [str(f[field])], "difficulty": 4, "field": field,
                    "distractors": [x['id'] for x in by_cluster[e['cluster']] if x['id'] != e['id']][:5]})
        i += 1
    # L5: aggregation over a set, 60
    i = 0
    for etype in ("client", "vendor", "project"):
        es = by_type[etype]
        for mid, op, field, q in AGG_METRICS:
            if i >= 60:
                break
            fvals = [(e, facts[e['id']]) for e in es]
            val = agg_value(fvals, (op, field))
            keys = []
            gold = set()
            # gold chunks: entity chunks holding the source field(s)
            if field == 'incident_severity:critical' or field == 'incident_severity:high':
                flds = [ff for ff in FIELDS if ff[0].endswith('_severity')]
            elif field == 'incident_duration_h':
                flds = [ff for ff in FIELDS if ff[0].endswith('_duration_h')]
            else:
                flds = [ff for ff in FIELDS if ff[0] == field.split(':')[0]]
            for e in es:
                for fdef in flds:
                    cid = fact_chunk.get(f"{e['id']}:{fdef[0]}")
                    if cid:
                        gold.add(cid)
            keys.append(str(val))
            if op in ('max', 'min') and ':' not in field and not field.startswith('incident'):
                # also the name of the extreme entity (dates compare as ISO strings)
                def _key(eid):
                    v = facts[eid][field]
                    return float(v) if _is_num(field) else str(v)
                nums = sorted([(_key(e['id']), e['name']) for e in es])
                keys.append(nums[-1][1] if op == 'max' else nums[0][1])
            out.append({"query_id": f"L5-{i + 1:03d}", "level": "L5",
                        "text": f"{q} среди всех {TYPE_RU[etype]}ов" if etype == 'project' else
                                f"{q} всех {TYPE_RU[etype]}ов",
                        "entity_ids": [e['id'] for e in es],
                        "gold_chunk_ids": sorted(gold),
                        "answer_keys": keys, "difficulty": 5,
                        "agg": {"type": etype, "metric": mid}})
            i += 1
    write_jsonl(BASE / 'queries' / 'queries.jsonl', out)
    print(json.dumps({"queries": len(out),
                      "by_level": {l: sum(1 for x in out if x['level'] == l) for l in 'L1L2L3L4L5'},
                      "para_llm_calls": stats.calls,
                      "para_prompt_tokens": stats.prompt_tokens,
                      "para_completion_tokens": stats.completion_tokens}, ensure_ascii=False))


# ---------------- index ----------------
def idx_dir(variant, scale):
    return BASE / 'results' / f'{variant}_{scale}' / 'index'


def self_ram_kb():
    try:
        for line in open('/proc/self/status'):
            if line.startswith('VmPeak:'):
                return int(line.split()[1])
    except Exception:
        pass
    return 0


def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def dir_size(p):
    total = 0
    for root, _dirs, files in os.walk(p):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


class RamSampler(threading.Thread):
    """Sample host RAM (KB) of qdrant + embed llama-server every 5s via ssh."""
    def __init__(self, out_csv):
        super().__init__(daemon=True)
        self.out_csv = out_csv
        self.samples = []
        self.stop_evt = threading.Event()
        self.pids = {}

    def resolve_pids(self):
        try:
            r = subprocess.run(HOST_SSH + ["sh", "-c",
                                           "echo Q=$(pgrep -x qdrant | head -1); echo E=$(pgrep -f 'bge-[m]3-Q4_K_M' | head -1)"],
                               capture_output=True, text=True, timeout=10)
            for line in r.stdout.splitlines():
                if line.startswith('Q=') and line[2:].strip():
                    self.pids['qdrant'] = line[2:].strip()
                if line.startswith('E=') and line[2:].strip():
                    self.pids['embed'] = line[2:].strip()
        except Exception:
            pass

    def run(self):
        self.resolve_pids()
        if not self.pids:
            return
        with open(self.out_csv, 'w') as f:
            f.write('t,qdrant_kb,embed_kb\n')
            t0 = time.time()
            while not self.stop_evt.is_set():
                try:
                    vals = []
                    for pid in (self.pids.get('qdrant'), self.pids.get('embed')):
                        if not pid:
                            vals.append('')
                            continue
                        r = subprocess.run(HOST_SSH + ["ps", "-o", "rss=", "-p", pid],
                                           capture_output=True, text=True, timeout=15)
                        v = [x for x in r.stdout.split() if x.strip().isdigit()]
                        vals.append(v[0] if v else '')
                    row = vals
                    self.samples.append((time.time() - t0, row))
                    f.write(f"{time.time() - t0:.1f},{row[0]},{row[1]}\n")
                    f.flush()
                except Exception:
                    pass
                self.stop_evt.wait(5)

    def peaks(self):
        res = {}
        for name, i in (('qdrant_ram_peak_kb', 0), ('embed_ram_peak_kb', 1)):
            vals = [s[1][i] for s in self.samples if s[1][i]]
            res[name] = max(vals) if vals else None
        return res


def qdrant_collection(name):
    r = requests.get(f"{QDRANT}/collections", timeout=10).json()
    return name in r['result']['collections']


def get_embed_model():
    try:
        names = [m['id'] for m in requests.get(f"{EMBED}/models", timeout=10).json()['data']]
        for c in EMBED_MODEL_CANDIDATES:
            if c in names:
                return c
        return names[0] if names else EMBED_MODEL_CANDIDATES[0]
    except Exception:
        return EMBED_MODEL_CANDIDATES[0]


def embed_batch(texts, model, stats, retries=5):
    # bge-m3 runs with -ub 1024 tokens (~4700 chars worst case): truncate long inputs
    # for the vector only (payload text in the index stays full).
    def cut(t):
        return t if len(t) <= 4000 else t[:4000]
    in_texts = [cut(t) for t in texts]
    for a in range(retries):
        stats['requests'] += 1
        t0 = time.time()
        try:
            r = requests.post(f"{EMBED}/embeddings", json={"model": model, "input": in_texts}, timeout=600)
            r.raise_for_status()
            stats['wall'] += time.time() - t0
            stats['tokens'] += sum(len(t) for t in in_texts) // 4
            return [d['embedding'] for d in r.json()['data']]
        except Exception:
            if a + 1 == retries:
                raise
            time.sleep(3 * (a + 1))


def idx_file(args, chunks, log):
    d = idx_dir(args.variant, args.scale)
    ents_dir = d / 'entities'
    ents_dir.mkdir(parents=True, exist_ok=True)
    by_ent = {}
    for c in chunks:
        by_ent.setdefault(c['entity_id'], []).append(c)
    for eid, cs in by_ent.items():
        text = "\n\n".join(c['text'] for c in sorted(cs, key=lambda x: x['chunk_id']))
        (ents_dir / f"{eid}.md").write_text(text)
    t0 = time.time()
    from rank_bm25 import BM25Okapi
    corpus = [tokenize(c['text']) for c in chunks]
    bm25 = BM25Okapi(corpus)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / 'bm25.pkl', 'wb') as f:
        import pickle
        pickle.dump({'bm25': bm25, 'ids': [c['chunk_id'] for c in chunks]}, f)
    wall = time.time() - t0
    log(f'bm25 built in {wall:.1f}s, {len(chunks)} chunks, {len(list(ents_dir.glob("*.md")))} md files')
    return {"wall_s": round(wall, 2), "disk_bytes": dir_size(d),
            "embed_requests": 0, "embed_tokens": 0, "embed_wall_s": 0}


def idx_fts(args, chunks, log):
    d = idx_dir(args.variant, args.scale)
    d.mkdir(parents=True, exist_ok=True)
    import sqlite3
    dbp = d / f'fts_{args.scale}.db'
    if dbp.exists():
        dbp.unlink()
    t0 = time.time()
    con = sqlite3.connect(dbp)
    con.execute("CREATE VIRTUAL TABLE chunks USING fts5(cid UNINDEXED, text, stem, tokenize='porter unicode61')")
    rows = []
    for c in chunks:
        rows.append((c['chunk_id'], c['text'], ' '.join(stem_tokens(tokenize(c['text'])))))
    con.executemany("INSERT INTO chunks VALUES (?,?,?)", rows)
    con.commit()
    con.close()
    wall = time.time() - t0
    log(f'sqlite fts5 indexed {len(chunks)} chunks in {wall:.1f}s')
    return {"wall_s": round(wall, 2), "disk_bytes": dir_size(d),
            "embed_requests": 0, "embed_tokens": 0, "embed_wall_s": 0}


def idx_rag(args, chunks, log):
    d = idx_dir(args.variant, args.scale)
    d.mkdir(parents=True, exist_ok=True)
    col = f"ragbench_{args.scale}"
    model = get_embed_model()
    es = {"requests": 0, "tokens": 0, "wall": 0.0}
    # fresh collection (only ours)
    if qdrant_collection(col):
        requests.delete(f"{QDRANT}/collections/{col}", timeout=60)
        log(f'deleting existing collection {col}...')
        for _ in range(120):
            if not qdrant_collection(col):
                break
            time.sleep(1)
    # create (409 Conflict = async delete not finished yet)
    r = None
    for a in range(10):
        r = requests.put(f"{QDRANT}/collections/{col}", json={
            "vectors": {"size": EMBED_DIM, "distance": "Cosine"}}, timeout=60)
        if r.status_code != 409:
            break
        log(f'create retry {a + 1}: 409 (delete still running)...')
        time.sleep(3)
    r.raise_for_status()
    log(f'created collection {col} (model={model})')
    t0 = time.time()
    B = 32
    for i in range(0, len(chunks), B):
        batch = chunks[i:i + B]
        vecs = embed_batch([c['text'] for c in batch], model, es)
        pts = [{"id": sha_id(c['chunk_id']),
                "vector": v,
                "payload": {"chunk_id": c['chunk_id'], "entity_id": c['entity_id'],
                            "doc_type": c['doc_type'], "text": c['text']}}
               for c, v in zip(batch, vecs)]
        for a in range(4):
            r = requests.put(f"{QDRANT}/collections/{col}/points", json={"points": pts}, timeout=300)
            if r.ok:
                break
            log(f'upsert retry {a + 1}: {r.status_code} {r.text[:200]}')
            time.sleep(2)
        if not r.ok:
            raise RuntimeError('qdrant upsert failed')
        if (i // B) % 10 == 0:
            log(f'embedded {i + len(batch)}/{len(chunks)}')
    # wait for indexing
    for a in range(60):
        r = requests.post(f"{QDRANT}/collections/{col}/points/count", json={"exact": True}, timeout=30)
        if r.ok and r.json()['result']['count'] == len(chunks):
            break
        time.sleep(2)
    else:
        raise RuntimeError('collection count mismatch')
    wall = time.time() - t0
    log(f'qdrant indexed {len(chunks)} chunks in {wall:.1f}s; embed: {es["requests"]} req, '
        f'~{es["tokens"]} tok, {es["wall"]:.1f}s')
    return {"wall_s": round(wall, 2), "disk_bytes": dir_size(d),
            "embed_requests": es["requests"], "embed_tokens": es["tokens"],
            "embed_wall_s": round(es["wall"], 2),
            "embed_tok_per_s": round(es["tokens"] / es["wall"], 1) if es["wall"] > 0 else None}


def cmd_index(args):
    p = corpus_paths(args.scale)
    chunks = read_jsonl(p['chunks'])
    res_dir = BASE / 'results' / f'{args.variant}_{args.scale}'
    res_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []
    def log(m):
        log_lines.append(f'[{time.strftime("%H:%M:%S")}] {m}')
        print(m, flush=True)
    log(f'index variant={args.variant} scale={args.scale} chunks={len(chunks)}')
    sampler = RamSampler(res_dir / 'ram_index.csv')
    sampler.start()
    t_cpu0 = cpu_s()
    t0 = time.time()
    if args.variant == 'file':
        extra = idx_file(args, chunks, log)
    elif args.variant == 'fts':
        extra = idx_fts(args, chunks, log)
    else:
        extra = idx_rag(args, chunks, log)
    wall = time.time() - t0
    sampler.stop_evt.set()
    sampler.join(timeout=20)
    host = sampler.peaks()
    metrics = {"variant": args.variant, "scale": args.scale, "stage": "index",
               "chunks": len(chunks),
               "index_wall_s": round(wall, 2),
               "index_ram_peak_kb": max(self_ram_kb(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
               "index_cpu_s": round(cpu_s() - t_cpu0, 2),
               **extra, **host}
    (res_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    (res_dir / 'index.log').write_text('\n'.join(log_lines))
    print(json.dumps(metrics, ensure_ascii=False))


# ---------------- search ----------------
SEARCHERS = {}


def searcher_file(variant, scale):
    import pickle
    d = idx_dir(variant, scale)
    with open(d / 'bm25.pkl', 'rb') as f:
        data = pickle.load(f)
    bm25, ids = data['bm25'], data['ids']
    def search(q, k):
        scores = bm25.get_scores(tokenize(q))
        top = sorted(range(len(ids)), key=lambda i: -scores[i])[:k]
        return [ids[i] for i in top if scores[i] > 0]
    return search


def searcher_fts(variant, scale):
    import sqlite3
    d = idx_dir(variant, scale)
    con = sqlite3.connect(d / f'fts_{scale}.db')
    def search(q, k):
        toks = tokenize(q)
        # drop English stop-words: OR-join of "the/who/is" matches half the corpus and kills ranking
        toks = [t for t in toks if t not in EN_STOP]
        terms = toks[:10] + stem_tokens(toks[:10])
        # keep only [a-z0-9] tokens: FTS5 chokes on quotes/parens (EN text)
        terms = [t for t in dict.fromkeys(terms) if len(t) > 1 and t.isalnum()]
        if not terms:
            return []
        stmt = ("SELECT cid FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?")
        rows = con.execute(stmt, (' OR '.join(terms), k)).fetchall()
        return [r[0] for r in rows]
    return search


def searcher_rag(variant, scale):
    col = f"ragbench_{scale}"
    model = get_embed_model()
    def search(q, k):
        v = embed_batch([q], model, {"requests": 0, "tokens": 0, "wall": 0})[0]
        r = requests.post(f"{QDRANT}/collections/{col}/points/search",
                          json={"vector": v, "limit": k, "with_payload": ["chunk_id"],
                                "with_vectors": False},
                          timeout=120)
        r.raise_for_status()
        return [p['payload']['chunk_id'] for p in r.json()['result']]
    return search


def cmd_search(args):
    variant, scale = args.variant, args.scale
    p = corpus_paths(scale)
    res_dir = BASE / 'results' / f'{variant}_{scale}'
    res_dir.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(queries_path(scale))
    if args.limit:
        queries = queries[:args.limit]
    chunks = {c['chunk_id']: c for c in read_jsonl(p['chunks'])}
    search = {'file': searcher_file, 'fts': searcher_fts, 'rag': searcher_rag}[variant](variant, scale)
    sampler = RamSampler(res_dir / 'ram_search.csv')
    sampler.start()
    t0 = time.time()
    results = []
    for i, q in enumerate(queries):
        t = time.time()
        top = search(q['text'], args.k)
        lat = (time.time() - t) * 1000
        rank = None
        for r, cid in enumerate(top[:10], 1):
            if cid in set(q['gold_chunk_ids']):
                rank = r
                break
        results.append({"query_id": q['query_id'], "level": q['level'], "top_k": top,
                        "latency_ms": round(lat, 2), "gold_rank": rank})
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(queries)} done', flush=True)
    sampler.stop_evt.set()
    sampler.join(timeout=20)

    # metrics
    def rel_metrics(subset):
        n = len(subset)
        out = {'n': n}
        for k in (1, 3, 5, 10):
            hits = sum(1 for r in subset if r['gold_rank'] is not None and r['gold_rank'] <= k)
            out[f'hit@{k}'] = round(hits / n, 4) if n else None
        out['mrr@10'] = round(sum(1 / r['gold_rank'] for r in subset
                                  if r['gold_rank'] is not None) / n, 4) if n else None
        ov = []
        for r in subset:
            q = next(x for x in queries if x['query_id'] == r['query_id'])
            keys = set()
            for a in q['answer_keys']:
                keys |= set(tokenize(a))
            if not keys:
                continue
            acc = 0.0
            for cid in r['top_k'][:5]:
                c = chunks.get(cid)
                if c:
                    acc += len(set(tokenize(c['text'])) & keys) / len(keys)
            ov.append(acc / 5)
        out['overlap@5'] = round(statistics.fmean(ov), 4) if ov else None
        return out

    lats = [r['latency_ms'] for r in results]
    lats.sort()
    def pct(p):
        if not lats:
            return None
        idx = min(len(lats) - 1, int(round(p / 100 * (len(lats) - 1))))
        return round(lats[idx], 2)

    by_level = {}
    for lv in ('L1', 'L2', 'L3', 'L4', 'L5'):
        sub = [r for r in results if r['level'] == lv]
        if sub:
            by_level[lv] = rel_metrics(sub)
    metrics = {"variant": variant, "scale": scale, "stage": "search", "k": args.k,
               "n_queries": len(results),
               "search_wall_s": round(time.time() - t0, 2),
               "latency_p50_ms": pct(50), "latency_p95_ms": pct(95), "latency_p99_ms": pct(99),
               "search_ram_peak_kb": max(self_ram_kb(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
               "host_ram_peak": sampler.peaks(),
               "overall": rel_metrics(results),
               "by_level": by_level}
    write_jsonl(res_dir / 'search_results.jsonl', results)
    mfile = res_dir / 'metrics.json'
    if mfile.exists():
        m = json.loads(mfile.read_text())
        m.update(metrics)
        metrics = m
    mfile.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps({"latency": {"p50": metrics['latency_p50_ms'], "p95": metrics['latency_p95_ms'],
                                  "p99": metrics['latency_p99_ms']},
                      "overall": metrics['overall']}, ensure_ascii=False))


# ---------------- judge (LLM e2e, rag+fts only) ----------------
def cmd_judge(args):
    variant, scale = args.variant, args.scale
    res_dir = BASE / 'results' / f'{variant}_{scale}'
    sr = {r['query_id']: r for r in read_jsonl(res_dir / 'search_results.jsonl')}
    queries = read_jsonl(queries_path(scale))
    chunks = {c['chunk_id']: c for c in read_jsonl(corpus_paths(scale)['chunks'])}
    # stratified sample: 20% -> 12 per level
    by_level = {}
    for q in queries:
        by_level.setdefault(q['level'], []).append(q)
    sample = []
    for lv in ('L1', 'L2', 'L3', 'L4', 'L5'):
        sample += by_level[lv][:12]
    search = {'rag': searcher_rag, 'fts': searcher_fts}[variant](variant, scale)
    stats = LLMStats()
    rows = []
    lats = []
    for i, q in enumerate(sample):
        t0 = time.time()
        top = search(q['text'], 5)
        ctx = "\n\n---\n\n".join((chunks.get(cid) or {'text': ''})['text'][:1200] for cid in top)
        ans = llm_chat([{"role": "system", "content": "Отвечай кратко (1-3 предложения) только по тексту контекста, с точными значениями."},
                        {"role": "user", "content": f"Контекст:\n{ctx}\n\nВопрос: {q['text']}"}],
                       temperature=0.2, max_tokens=200, stats=stats)
        judge = llm_chat([{"role": "user", "content":
                           f"Проверь, содержит ли ответ все требуемые значения.\nВопрос: {q['text']}\nОтвет: {ans}\n"
                           f"Требуемые значения: {json.dumps(q['answer_keys'], ensure_ascii=False)}\n"
                           "Верни JSON {\"correct\": true/false, \"missing\": \"...\"}."}],
                         temperature=0.1, max_tokens=150, stats=stats)
        try:
            jd = extract_json(judge)
            correct = bool(jd.get('correct'))
        except Exception:
            correct = any(str(a) in ans for a in q['answer_keys'])
        lat = time.time() - t0
        lats.append(lat * 1000)
        rows.append({"query_id": q['query_id'], "level": q['level'], "answer": ans,
                     "correct": correct, "e2e_latency_ms": round(lat * 1000, 1)})
        if (i + 1) % 10 == 0:
            print(f'  judge {i + 1}/{len(sample)}', flush=True)
    lats.sort()
    def pct(p):
        return round(lats[min(len(lats) - 1, int(round(p / 100 * (len(lats) - 1))))], 1)
    metrics = {"judge": {"n": len(rows),
                         "e2e_correct_rate": round(sum(1 for r in rows if r['correct']) / len(rows), 4),
                         "e2e_p50_ms": pct(50), "e2e_p95_ms": pct(95),
                         "llm_calls": stats.calls,
                         "llm_prompt_tokens": stats.prompt_tokens,
                         "llm_completion_tokens": stats.completion_tokens}}
    mfile = res_dir / 'metrics.json'
    m = json.loads(mfile.read_text()) if mfile.exists() else {}
    m.update(metrics)
    mfile.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    write_jsonl(res_dir / 'judge.jsonl', rows)
    print(json.dumps(metrics, ensure_ascii=False))


# ---------------- run-all ----------------
def cmd_run_all(args):
    ns = argparse.Namespace(variant=args.variant, scale=args.scale, k=10, limit=None)
    cmd_index(ns)
    cmd_search(ns)
    if args.judge and args.variant in ('rag', 'fts'):
        cmd_judge(argparse.Namespace(variant=args.variant, scale=args.scale, sample=60))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sp = sub.add_parser('gen-corpus'); sp.add_argument('--scale', required=True); sp.add_argument('--limit', type=int)
    sp.add_argument('--force', action='store_true'); sp.set_defaults(fn=cmd_gen_corpus)
    sp = sub.add_parser('gen-queries'); sp.add_argument('--scale', default='1k')
    sp.add_argument('--force', action='store_true'); sp.set_defaults(fn=cmd_gen_queries)
    sp = sub.add_parser('index'); sp.add_argument('--variant', required=True, choices=['file', 'fts', 'rag'])
    sp.add_argument('--scale', required=True); sp.set_defaults(fn=cmd_index)
    sp = sub.add_parser('search'); sp.add_argument('--variant', required=True, choices=['file', 'fts', 'rag'])
    sp.add_argument('--scale', required=True); sp.add_argument('--k', type=int, default=10)
    sp.add_argument('--limit', type=int); sp.set_defaults(fn=cmd_search)
    sp = sub.add_parser('judge'); sp.add_argument('--variant', required=True, choices=['rag', 'fts'])
    sp.add_argument('--scale', required=True); sp.add_argument('--sample', type=int, default=60)
    sp.set_defaults(fn=cmd_judge)
    sp = sub.add_parser('run-all'); sp.add_argument('--variant', required=True, choices=['file', 'fts', 'rag'])
    sp.add_argument('--scale', required=True); sp.add_argument('--judge', action='store_true')
    sp.set_defaults(fn=cmd_run_all)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
