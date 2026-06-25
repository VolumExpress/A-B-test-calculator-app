# Banking Experiment Calculator

Streamlit-приложение для проектирования, мониторинга и анализа банковских пилотов и причинных экспериментов.

Приложение сочетает:

- классический A/B и multi-arm дизайн;
- анализ уже запущенного пилота;
- Bayesian и sequential monitoring;
- variance reduction через CUPED/CUPAC;
- survival и event-history методы;
- uplift, policy value и Next Best Action;
- квазиэкспериментальные дизайны, когда обычная рандомизация невозможна;
- выгрузку результатов в Excel, HTML и PDF.

> **Статус:** расширенный MVP. Приложение предназначено для аналитического проектирования и первичной оценки пилотов. Оно не заменяет независимую модельную валидацию, Model Risk Management, юридическое согласование и регуляторные процедуры.

---

## Что реализовано

### 1. Проектирование классического пилота

Пошаговый мастер позволяет:

- заполнить паспорт пилота;
- описать бизнес-гипотезу и фактическое воздействие;
- выбрать бинарную, непрерывную или multi-arm метрику;
- задать baseline, MDE, alpha, power и доступный трафик;
- рассчитать выборку и срок;
- оценить эффект неравного распределения;
- построить сценарии сокращения срока;
- подготовить O'Brien-Fleming-подобный interim-план;
- выгрузить паспорт и расчёты в Excel.

### 2. Анализ A/B и multi-arm результатов

Поддерживаются:

- CSV, XLS и XLSX;
- бинарные и непрерывные outcome;
- Z-test двух долей;
- Fisher exact test;
- Welch t-test;
- доверительные интервалы;
- относительный и абсолютный эффект;
- Sample Ratio Mismatch;
- Holm, Bonferroni и Benjamini-Hochberg коррекции;
- базовая uplift-калибровка;
- Excel-отчёт.

### 3. Bayesian monitoring

Реализованы:

- Beta-Binomial анализ двух групп;
- Jeffreys prior;
- исторический prior через mean + effective sample size;
- prior-predictive проверка;
- sensitivity analysis по нескольким priors;
- вероятность, что treatment лучше;
- вероятность достижения заданного минимального эффекта;
- credible intervals абсолютного и относительного эффекта;
- predictive probability успеха после добора максимальной выборки;
- Monte Carlo standard error predictive probability.

### 4. Sequential и exact-sequential

Реализованы:

- O'Brien-Fleming и Pocock границы;
- симуляционная Gaussian calibration;
- анализ накопленных interim-результатов;
- conditional power;
- non-binding futility;
- exact-sequential calibration для Fisher, Boschloo и Barnard;
- симуляционный контроль общего Type I error при повторных просмотрах;
- применение calibrated exact threshold к накопленному пути пилота.

Тяжёлая exact-калибровка запускается в отдельном процессе.

### 5. CUPED и CUPAC по фактическим данным

Реализованы:

- CUPED/ANCOVA по одному или нескольким pre-period признакам;
- robust HC3 standard errors;
- CUPAC с out-of-fold прогнозом outcome;
- cross-fitting;
- числовые и категориальные признаки;
- оценка реального снижения дисперсии;
- оценка sample-size multiplier;
- предупреждение, если корректировка увеличивает дисперсию.

> Используйте только признаки, сформированные до назначения treatment.

### 6. Survival, RMST и non-proportional hazards

Реализованы:

- Kaplan-Meier curves;
- log-rank test;
- early- и late-weighted log-rank;
- Cox proportional hazards;
- тест proportional hazards;
- RMST до выбранного горизонта;
- bootstrap CI разности RMST;
- milestone survival;
- предупреждение при non-proportional hazards.

При нарушении PH приложение рекомендует делать главным выводом RMST или milestone, а не единый hazard ratio.

### 7. Competing risks

Реализованы:

- Aalen-Johansen cumulative incidence;
- CIF по группам;
- bootstrap CI разности CIF на горизонте;
- cause-specific Cox;
- отдельное отображение конкурирующих событий.

### 8. Recurrent events

Реализованы:

- Andersen-Gill counting-process Cox;
- cluster-robust sandwich SE по субъекту;
- Poisson GEE;
- Negative Binomial GEE;
- rate ratio и hazard ratio.

### 9. Кластерные дизайны

Реализованы:

- design effect с поправкой на ICC;
- поправка на неодинаковый размер кластеров;
- inflation на attrition;
- расчёт примерного числа кластеров;
- генерация stepped-wedge schedule;
- генерация switchback schedule;
- block length и washout;
- GEE-анализ cluster-period данных;
- period fixed effects;
- carryover covariate.

### 10. Квазиэкспериментальные методы

#### Synthetic control

- неотрицательные donor weights;
- сумма весов равна 1;
- pre-period RMSE;
- post-intervention gap;
- placebo donor analysis.

#### Regression discontinuity

- sharp и fuzzy RDD;
- local linear / quadratic regression;
- triangular kernel;
- bandwidth;
- reduced-form jump;
- first-stage jump;
- local treatment effect.

#### Continuous treatment / dose-response

- spline outcome regression;
- adjustment по pre-treatment covariates;
- bootstrap confidence bands;
- поиск лучшей точки на наблюдаемой dose grid.

### 11. Ranking и contextual bandits

#### Interleaving

- победитель A/B/tie по сессии;
- exact binomial test;
- bootstrap confidence interval;
- cluster bootstrap по пользователю.

#### Contextual bandits

- IPS;
- SNIPS;
- doubly robust estimator;
- deterministic и stochastic target policy;
- effective sample size importance weights;
- overlap diagnostics;
- bootstrap confidence intervals.

### 12. Uplift и policy evaluation

#### Qini / AUUC

- IPW uplift curve;
- Qini coefficient;
- AUUC;
- bootstrap confidence intervals;
- cluster bootstrap;
- uplift calibration by bins;
- predicted versus observed uplift.

#### Doubly robust policy value

- multi-action policy evaluation;
- cross-fit propensity model;
- cross-fit outcome models;
- AIPW/DR value;
- IPS comparison;
- policy match rate;
- effective sample size weights;
- bootstrap confidence interval.

#### Capacity-aware Next Best Action

- несколько действий и no-action baseline;
- не более одного действия на клиента;
- capacity на каждый канал;
- value и cost columns;
- contact-fatigue penalty;
- MILP для умеренных объёмов;
- greedy fallback для больших файлов;
- выгрузка клиентских назначений в CSV.

### 13. Отчёты

Результаты текущей сессии можно объединить в:

- Excel workbook;
- HTML-протокол;
- PDF-протокол.

Отчёт включает паспорт, табличные результаты, предупреждения и ограничения.

---

## Для каких банковских задач подходит

### Хорошее покрытие

| Задача | Примеры |
|---|---|
| Churn и удержание | звонок, оффер, новая политика удержания |
| Продажи и propensity | cross-sell, up-sell, выбор клиентов |
| Коммуникации | SMS, push, email, звонок, multi-arm |
| Uplift и NBA | Qini/AUUC, policy value, capacity constraints |
| CLTV-политики | выбор клиентов для воздействия |
| Рекомендации и ranking | A/B и interleaving |
| Коллекшн | survival, recurrent events, cluster/operator effects |
| AI-ассистенты | время, качество, конверсия, repeated observations |
| Операционные процессы | cluster-period, stepped-wedge, switchback |
| Pricing и лимиты | dose-response внутри допустимой области |
| Пороговые политики | regression discontinuity |
| Внедрение без рандомизации | synthetic control |
| Contextual bandits | offline IPS/SNIPS/DR evaluation |

### Подходит с существенными оговорками

| Задача | Ограничение |
|---|---|
| Fraud | adversarial adaptation, graph interference и delayed labels требуют отдельного дизайна |
| AML | неполные labels и investigator feedback loop |
| Кредитный андеррайтинг | тест только внутри риск-аппетита и разрешённой политики |
| Персонализированный pricing | требуется проверка fairness, monotonicity и legal constraints |
| Гео- и сетевые эффекты | возможен spillover между единицами рандомизации |
| Очень большие MILP | приложение переключается на greedy approximation |

### Не является основной валидацией

Приложение не заменяет специализированную валидацию для:

- PD, LGD, EAD, CCF;
- IFRS 9 и резервирования;
- VaR, Expected Shortfall и XVA;
- ALM и ликвидности;
- ICAAP и стресс-тестирования;
- чистых forecasting-моделей без управляющего воздействия;
- макроэкономических моделей;
- полной безопасности и compliance-валидации Generative AI.

---

## Режимы приложения

### Проектирование пилота

Простой пошаговый мастер для стандартного A/B или multi-arm.

### Анализ результатов

Загрузка итогового CSV/XLSX и классический статистический анализ.

### Расширенные методы

Отдельный selector содержит:

1. Bayesian monitoring;
2. Sequential и exact-sequential;
3. CUPED/CUPAC;
4. Survival/RMST/non-PH;
5. Competing risks;
6. Recurrent events;
7. Cluster/stepped-wedge/switchback;
8. Synthetic control;
9. RDD;
10. Interleaving;
11. Dose-response;
12. Contextual bandits;
13. Qini/AUUC;
14. DR policy value;
15. Capacity-aware NBA;
16. HTML/PDF/Excel protocol.

---

## Форматы входных данных

В `assets/advanced_input_templates.xlsx` находятся отдельные листы с примерами для всех продвинутых методов.

### Sequential

```text
x_control | n_control | x_treatment | n_treatment
```

Одна строка — один накопленный interim-анализ.

### CUPED/CUPAC

```text
client_id | treatment | outcome | preperiod_metric | risk_score | segment
```

### Survival

```text
client_id | group | duration | event
```

`event=0` — цензурирование, `event=1` — событие.

### Competing risks

```text
client_id | group | duration | event_type
```

`event_type=0` — цензурирование; `1` — событие интереса; `2+` — competing events.

### Recurrent events

```text
client_id | start | stop | event | treatment
```

### Cluster-period

```text
cluster | period | treatment | outcome | carryover
```

### Synthetic control

```text
unit | time | outcome
```

### Regression discontinuity

```text
client_id | running_variable | outcome | actual_treatment
```

### Uplift

```text
client_id | treatment | outcome | predicted_uplift | propensity | cluster_id
```

### Policy value

```text
client_id | observed_action | target_action | reward | behavior_propensity | features...
```

### Contextual bandits

```text
client_id | logged_action | target_action | reward | behavior_propensity | q_action_1 | q_action_2 ...
```

### Capacity-aware NBA

```text
client_id | value_no_action | value_sms | value_push | value_call | contact_fatigue
```

---

## Установка

Требуется Python 3.11+.

### Windows

```bat
run_windows.bat
```

### Linux / macOS

```bash
chmod +x run_linux_mac.sh
./run_linux_mac.sh
```

### Ручной запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows activation:

```bat
.venv\Scripts\activate
```

Открыть:

```text
http://localhost:8501
```

---

## Docker

```bash
docker build -t banking-experiment-calculator .
docker run --rm -p 8501:8501 banking-experiment-calculator
```

---

## Тесты

```bash
pytest -q
```

Тесты покрывают:

- классический дизайн и анализ;
- Bayesian и predictive probability;
- Gaussian и exact sequential;
- CUPED/CUPAC;
- survival, competing и recurrent events;
- cluster, RDD, synthetic control и dose-response;
- Qini/AUUC, DR policy value, bandits и NBA;
- HTML, PDF и Excel;
- smoke test Streamlit-мастера.

---

## Структура проекта

```text
banking_experiment_mvp/
├── app.py
├── advanced_ui.py
├── experiment_core/
│   ├── analysis.py
│   ├── design.py
│   ├── bayesian.py
│   ├── sequential.py
│   ├── variance_reduction.py
│   ├── survival.py
│   ├── causal_designs.py
│   ├── uplift_advanced.py
│   ├── bandits_ranking.py
│   ├── background.py
│   ├── excel_report.py
│   └── reporting.py
├── assets/
│   ├── pilot_data_template.xlsx
│   ├── pilot_sample_data.xlsx
│   └── advanced_input_templates.xlsx
├── notebooks/
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Архитектурные ограничения MVP

### Нет базы данных

- данные и результаты живут в текущей Streamlit-сессии;
- после рестарта сервера они не восстанавливаются;
- для сохранения используйте Excel/HTML/PDF.

### Фоновые задачи

- exact и тяжёлый uplift bootstrap могут выполняться в отдельном процессе;
- очередь хранится только в памяти приложения;
- нет Celery/RQ, persistent queue и recovery после рестарта;
- максимальное число worker-процессов ограничено.

### Методологические ограничения

- causal assumptions не проверяются автоматически полностью;
- synthetic control не устраняет одновременные внешние шоки;
- RDD требует отсутствия манипулирования около cutoff;
- dose-response зависит от observed confounders и overlap;
- DR не спасает при одновременной ошибке propensity и outcome model;
- contextual bandit evaluation нестабилен при больших weights;
- Qini/AUUC должны считаться на независимых или out-of-fold predictions;
- switchback требует достаточного washout;
- stepped-wedge чувствителен к временным трендам;
- competing-risk методы отвечают на разные estimands;
- HR не должен быть единственным выводом при non-PH.

---

## Безопасность данных

Не загружайте:

- ФИО;
- телефоны;
- email;
- номера счетов и договоров;
- паспортные данные;
- иные прямые идентификаторы.

Используйте обезличенные технические идентификаторы.

---

## Что логично добавить после MVP

- SSO и роли;
- persistent job queue;
- хранение версий протокола;
- audit log;
- полноценные stochastic target policies в UI;
- generalized random forests для CATE;
- TMLE;
- Gray test и Fine-Gray regression;
- randomization inference для кластерных дизайнов;
- geo-experiments;
- network interference;
- fairness constraints в NBA/pricing;
- отдельный контур model validation для PD/LGD/EAD и forecasting.

---

## Дисклеймер

Результаты приложения являются аналитической поддержкой решения. Перед промышленным внедрением необходимо проверить:

- качество и происхождение данных;
- корректность единицы рандомизации;
- соответствие предпосылкам метода;
- правила остановки;
- guardrail-метрики;
- юридические ограничения;
- риск-аппетит;
- независимую валидацию.
