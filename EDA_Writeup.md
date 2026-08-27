# My AQI Predictor Journey (EDA + Everything I Did)

This is basically the story of how I built this whole project — which API I picked and why, how I cleaned and prepped the data, what features I made, which models I tried and how they actually performed, why I picked the final one, and all the annoying problems I hit along the way (and how I got past them). I'm writing it in plain language, the way I'd actually explain it to a friend, not like a textbook.

---

## 1. Picking the API

The brief said I could use AQICN or OpenWeather, or something else. I ended up going with **OpenWeather** for two reasons:

- One API key gets me both the weather data (temperature, humidity, wind speed) AND the pollution data (PM2.5, PM10, CO, NO2, SO2, O3) I needed. Two different things from one place.
- It has a proper historical data endpoint too, which I thought I'd need for backfilling old data.

I also picked coordinates (Karachi is 24.8607°N, 67.0011°E) instead of just typing "Karachi" as a city name, because city names can be ambiguous — there could be multiple places with similar names — but coordinates are exact, no confusion.

Later on, when I needed to backfill three years of historical data, OpenWeather's historical access turned out to be limited on the free plan, so I switched to **Open-Meteo** just for that one-time backfill job. It's free and has a solid historical archive. So the final setup is: OpenWeather for the live hourly stuff, Open-Meteo for the one-time historical dump. Both feed into the exact same feature format so it doesn't matter which one a row came from once it's processed.

One thing I want to point out because I was actually proud I caught this: **I didn't use the AQI number the API gives you.** OpenWeather returns its own AQI, but it's on a 1-to-5 scale, not the real 0-500 scale that AQI is supposed to be (the one you see on actual air quality apps and news). So I wrote my own function that takes the raw PM2.5 and PM10 numbers and runs them through the official EPA's math (it's called a "breakpoint" formula — basically a table that tells you how to convert pollution concentration into an AQI score, with some interpolation in between the table's fixed points). I did this the same way for both the live data and the historical backfill so the AQI number would be consistent no matter which API it came from. If I'd used two different scales for training vs live data, the model would've learned garbage.

Also, my API key isn't sitting in my code anywhere — it lives in a `.env` file that's in `.gitignore`, so it never gets pushed to GitHub. Basic thing, but easy to mess up if you're not careful, and I didn't want a public repo with my key just sitting there for anyone to grab.

---

## 2. Cleaning the data

Cleaning here wasn't like "removing weird typos from a spreadsheet" — it was more about handling gaps and making sure I wasn't accidentally feeding the model broken or incomplete rows.

A few things I had to deal with:

- **Rows with missing future data.** Since each row's "target" is the average AQI over the *next* 24/48/72 hours, if a chunk of those future hours is missing (say the API had a hiccup and there's a gap), you can't just average the 3 values you do have and pretend that's a full day's average. So I set a rule: if more than half the hours in that future window are missing, just drop the target for that row entirely instead of faking it.
- **Rows without enough history.** Same idea but backwards — features like "AQI 24 hours ago" or the rolling 24-hour average need enough past data to actually exist. Rows near the very start of the dataset (or right after a gap) don't have that, so those get dropped too.
- **Chronological split, not random.** When I split into training data and testing data, I didn't shuffle it randomly like you'd normally do. I took the oldest 80% for training and the newest 20% for testing, in time order. If I'd shuffled randomly, the model could "see the future" during training (a row from next week ending up in training while a row from last week is in testing), which would make my test results look way better than they actually are. This is called data leakage and it's a really easy mistake to make without realizing it.
- **Nearest-timestamp matching for live data.** When the live dashboard needs "AQI 24 hours ago" for a fresh prediction, it might not have an exact timestamp match (maybe there was a small gap in data collection). So instead of crashing or failing, it finds the *closest* available timestamp and uses that, and it also tracks how far off that match actually was, so it's honest about it rather than silently guessing.

After all that cleaning, I ended up with about **26,172 usable rows** out of the ~26,300 I originally collected — so I only lost a small chunk to gaps, which felt like a good sign the pipeline was actually reliable.

---

## 3. Feature engineering

This is where I turned raw numbers into stuff the model could actually learn patterns from. Two categories:

**Time-based features** — hour of day, day of the month, month, and day of the week. AQI has patterns tied to time (like worse air at certain hours or seasons), so the model needs to see time as an actual input, not just guess it.

**Trend/derived features** — this is the part that took the most thinking:
- AQI from 24, 48, and 72 hours ago (lag features)
- How much AQI changed in the last 24 hours
- A rolling 24-hour average and a rolling 24-hour standard deviation (basically: "has it been calm or jumping around lately")

These matter because AQI doesn't just depend on right-now conditions, it depends on the recent trend. A reading of 80 that's been climbing for two days means something different than a reading of 80 that's been flat.

### The big redesign: what am I actually predicting?

Originally I built this to predict one single number — "the AQI exactly 72 hours from now." After talking to my mentor and the rest of the cohort, I found out that's not really how AQI forecasts are supposed to work. Real forecasts report a **day's average**, not one random instant. So I rebuilt it as **three separate models**:

- Model 1 predicts the average AQI over hours 1–24 (basically "tomorrow")
- Model 2 predicts hours 25–48 ("the day after")
- Model 3 predicts hours 49–72 ("three days out")

Each one is trained completely separately on the same features, rather than chaining them (like using Day 1's prediction as an input to predict Day 2). A classmate actually tried the chaining approach and found that errors from Day 1 just pile up and make Day 2 and Day 3 way worse. So I stuck with three independent models — more honest, and it matched what other people in the cohort were also landing on.

---

## 4. Models I tried, and how they actually did

I tested five different models across all three time horizons, using the same train/test split for a fair comparison: **Ridge Regression**, **Random Forest**, **Gradient Boosting**, **XGBoost**, and (later on) an **LSTM** (a deep learning model built for sequences/time series).

I judged them with three standard metrics:
- **MAE** — average error in plain AQI points, easy to read
- **RMSE** — similar to MAE but punishes big misses harder
- **R²** — roughly "how much better than just guessing the average" the model is (more on why this number is tricky below)

Here's the actual comparison table, straight from running the scripts:

| Horizon | Model | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| 24h | Persistence (baseline: "today = tomorrow") | 16.76 | – | 0.083 |
| 24h | Ridge | 11.49 | 8.26 | 0.569 |
| 24h | Random Forest | 12.08 | 8.64 | 0.523 |
| 24h | Gradient Boosting | 11.81 | 8.33 | 0.544 |
| 24h | XGBoost | 11.98 | 8.41 | 0.532 |
| 24h | **LSTM** | **11.18** | **8.15** | **0.592** |
| 48h | Persistence | 21.44 | – | −0.567 |
| 48h | Ridge | 14.48 | 10.78 | 0.285 |
| 48h | Random Forest | 15.96 | 11.73 | 0.132 |
| 48h | Gradient Boosting | 15.74 | 11.55 | 0.156 |
| 48h | XGBoost | 15.45 | 11.35 | 0.187 |
| 48h | LSTM | 14.54 | 10.83 | 0.280 |
| 72h | Persistence | 23.12 | – | −0.952 |
| 72h | Ridge | 15.02 | 11.18 | 0.176 |
| 72h | Random Forest | 16.93 | 12.74 | −0.047 |
| 72h | Gradient Boosting | 17.00 | 12.70 | −0.056 |
| 72h | XGBoost | 16.99 | 12.60 | −0.054 |
| 72h | **LSTM** | **14.73** | **11.08** | **0.208** |

A few things jump out looking at this:

- Every single one of my models beats "persistence" (just guessing tomorrow = today), which is honestly the bar that actually matters — I'll explain why in the next section.
- The tree-based models (Random Forest, Gradient Boosting, XGBoost) actually go **negative R²** at 72 hours — meaning they're doing *worse* than just guessing the average. That's overfitting: the further out you predict, the noisier and harder the target gets, and these models latched onto patterns in the training data that didn't generalize.
- Ridge (the simplest model here, just plain linear regression with a little regularization) held up the best out of the four "classic" models across all three horizons.
- The LSTM actually won on 24h and 72h, and came very close to Ridge on 48h. So deep learning did help a bit, especially at the longer horizon where the tree models fell apart.

---

## 5. How I picked the final model

Here's a big thing I learned that changed how I evaluated everything: **R² isn't the real bar here.** My mentor explained that R² gets mechanically squashed low whenever the test period happens to be a calm, stable stretch where AQI isn't moving around much — that's just math, it happens no matter how good your model actually is. So chasing "R² above some number like 0.70" is kind of a trap.

The actual bar that matters is: **does your model beat "persistence"?** Persistence just means "assume tomorrow looks exactly like today" — the dumbest possible forecast. If your model can't beat that, it's not adding any real value. Looking at the table above, Ridge (and every other model) crushes persistence at every horizon, so that box is checked.

Given that, I went with **Ridge Regression as the model that actually gets used in production** (the one that gets retrained daily and registered). Here's my actual reasoning, not just "it had decent numbers":

- It beat every other "classic" model (RF, GB, XGBoost) at all three horizons, and didn't fall apart at 72h like they did.
- The LSTM technically won on two out of three horizons, but only by a small amount, and it comes with real costs: it needs TensorFlow (a much heavier dependency), it needs the input data scaled and reshaped into sequences (more moving parts that can break), and it takes longer to retrain. My whole project retrains itself automatically every single day with no person watching it — for that kind of unattended, serverless setup, a small, fast, boringly-reliable model that's easy to reason about beats a slightly-more-accurate one that's more fragile and expensive to keep running. If the LSTM's win margin had been huge, I'd have made a different call. It wasn't.
- Ridge is also easy to explain. Since it's linear, I could hook up SHAP (explained more below) in an exact, no-approximation way, which matters for a project where showing "why did the model predict this" is part of the point.

So: I still built and evaluated the LSTM (which the brief explicitly asks for — "from statistical to deep learning models") and it's documented honestly here with real numbers, but Ridge stays the one actually running the show day to day.

---

## 6. Problems I ran into (and how I actually fixed them)

This is the part I think matters most, because most of what I actually learned came from things breaking.

**Windows + conda not talking to each other.** Right at the start, VS Code's terminal didn't know what `conda` even was. Turns out you have to run `conda init powershell` once from a separate Anaconda Prompt window, then open a *fresh* VS Code terminal for it to work. Small thing, but confusing the first time.

**A bunch of Windows-specific package install headaches with Hopsworks.** Getting the Hopsworks Python client fully working on Windows was genuinely annoying — I hit build failures on a package called `twofish`, issues with `pyarrow` (the library that actually talks to Hopsworks' data service), and Hopsworks code that assumed a Linux-style `/tmp` folder which Windows just doesn't have the same way. I worked through these one at a time, mostly by finding the right package versions and setting things up so Hopsworks had a temp folder it could actually use on Windows. Not going to pretend I remember the exact fix for every single one at this point, but I got through all of them and the pipeline has been running reliably since.

**Reading 26,000 rows from Hopsworks in one go was unreliable.** Early on, trying to pull the entire training dataset in a single request would randomly fail or hang. The fix was to split it into smaller time-windowed chunks (60 days at a time) and read them one after another, then glue the results together. Much more reliable, if slightly slower.

**The big one: recent data basically disappeared when I queried it.** Even though my hourly job was successfully sending data to Hopsworks (I could see hundreds of successful runs), if I tried to query anything from roughly the last one to two weeks, I'd get **zero rows back** — every time, no matter how I asked for it. Older data (like the 3-year historical backfill) worked totally fine. This one took real digging. Eventually I found the actual reason sitting in the logs the whole time: a warning saying *"Materialization job is already running... please wait for the current execution to finish."* Basically, Hopsworks needs a background job to process newly-inserted rows before they're actually queryable, and on the free tier that job just can't keep up with hourly inserts — it's permanently behind.

The fix: since my hourly script already has each record sitting in memory right before it sends it to Hopsworks anyway, I just made it *also* save a copy into a small local CSV file, and commit that file back to the GitHub repo every hour. So the live dashboard reads recent history from that local file instead of asking Hopsworks for it. Hopsworks is still the "real" feature store and training still reads from it (older data has had time to materialize by then) — this local file only exists to work around that one specific query limitation for very recent data.

**Two automated jobs pushing to GitHub at the same time.** My hourly pipeline and my daily training job both commit changes back to the repo. Sometimes they'd overlap and one push would get rejected because the other one had just changed something. Fixed by having each workflow do a `git pull --rebase` right before it pushes, so it grabs any new changes first instead of colliding.

**Streamlit Cloud deployment kept failing.** When I finally tried to deploy the dashboard publicly, the build kept failing with an error about a module called `imp` not existing. Took me two tries to actually fix. First I tried pinning the Python version with a `runtime.txt` file — didn't work, turns out Streamlit Cloud's newer installer just ignores that file. The real problem was that `hopsworks` (an old dependency) still uses very old install code that needs a Python module (`imp`) that got removed in newer Python versions. The actual fix: I realized the dashboard itself never even imports `hopsworks` — only my backend automation scripts do — so I just removed it from the shared requirements file the dashboard installs from, and added it separately only to the GitHub Actions workflows that actually need it. Dashboard deployed fine after that.

**A training run failed randomly with a "socket closed" error.** One day my automated daily training job just failed out of nowhere with a connection error while reading from Hopsworks. Looked into it and it was a one-off network hiccup on Hopsworks' side, not anything wrong with my code (the same script worked fine the day before and after). Added a small retry — if reading a chunk fails, wait a bit and try again a couple times before giving up — so a random blip like that doesn't kill the whole day's training run anymore.

**Other people's suspiciously perfect results.** A couple of times, classmates reported numbers that seemed way too good (like R² of 0.997, basically a "perfect" model). I looked at the actual numbers and they didn't make mathematical sense together for a 3-day-ahead AQI forecast — that combination of a tiny error and a near-perfect R² usually means the model accidentally got to "cheat" somehow (most commonly, a feature that leaks information from the future into training, or the data wasn't split in time order). I didn't chase copying their approach — I'd rather have honest numbers I understand than borrowed numbers I can't explain.

---

## 7. What's still left

Multi-city support (this whole thing is built just for Karachi right now) is the one thing on the original wishlist I haven't gotten to. Everything else from the brief — the pipeline, the automation, the dashboard, SHAP explanations, hazardous AQI alerts, and now this write-up plus the deep learning model comparison — is done.

---

*One honest note: I used AI help (Claude) quite a bit while building the dashboard's look and feel, since frontend/web stuff isn't really my area — that's already called out in the main report. Everything about the data, the features, the models, and the actual debugging in this write-up is my own work and understanding.*
