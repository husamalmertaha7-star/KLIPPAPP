# Klipp — working MVP (now with accounts + billing)

A real, runnable version of the "long video → short clips" engine your landing
page (ai-scene-snipper.lovable.app) advertises — now with real user accounts,
a pricing page, and Stripe checkout wired in, so people can actually sign up
and pay for Creator.

This was built from scratch in a sandbox with **no internet access** (no pip,
no npm, no OpenAI/Google/Stripe APIs, no model downloads). Everything below is
scoped honestly around that constraint — see "What could not be tested here"
before you assume something works.

## What actually works right now (tested end-to-end, offline)

**The clipping engine:**
- **Upload → background processing → download**, via a Flask app
  (`backend/app.py`) with a per-user job queue.
- **Hook detection without a transcript.** `audio_analysis.py` extracts the
  audio track and computes real signal features (loudness/RMS, speaking-pace
  variance, spectral flux for bursts/emphasis, silence via `ffmpeg
  silencedetect`). `highlight_scoring.py` slides a window across the video,
  scores it, and greedily picks the top non-overlapping moments, snapping cut
  points to nearby silence.
- **Internal silence trimming**, **face-aware auto-reframe** (OpenCV, falls
  back to center-crop with no face), and a **score badge overlay** burned
  into every export.
- Verified against a generated 90-second test video with two distinct loud
  sections — scored the louder one 100 vs. 71.6, picked correct non-overlapping
  clips, trimmed silence, rendered all three aspect ratios.

**Accounts + billing (new):**
- **Real signup/login** (`auth.py`) — email + password, hashed with
  werkzeug's PBKDF2, session-cookie based. No third-party auth needed.
- **Per-user SQLite database** (`db.py`, zero setup — it's a single file at
  `jobs/klipp.db`) tracking plan, Stripe customer/subscription IDs, and a
  rolling 30-day clip quota.
- **Plan gating that's actually enforced**, not cosmetic: Free accounts get
  3 clips per 30 days at 1080p; a job that would exceed the quota is either
  capped to however many clips are left, or rejected outright with a 402 and
  a clear upgrade message. If a job fails or produces fewer clips than
  reserved, the unused quota is refunded automatically. Creator/Studio get
  unlimited clips at 4K.
- **Stripe Checkout, webhook, and billing portal** (`billing.py`) for the
  Creator plan (€24/mo). Studio (€79/mo) is sales-assisted — like the
  original landing page's "Talk to us" button — and just records a contact
  request in the database rather than pretending to be self-serve.
- Three pages: `index.html` (marketing + pricing + sign up/sign in),
  `app.html` (the clipping tool, requires login), `account.html` (plan,
  usage bar, upgrade / manage-billing / sign out).

Every one of the bullets above under "Accounts + billing" was actually
exercised against the running server in this sandbox: real signup/login/
logout, duplicate-email rejection, session expiry, cross-user job access
correctly blocked (404), the full quota lifecycle (reserve → partial-allow →
refund → hard block at 402), and — since Stripe's *webhook signing* is just
HMAC-SHA256 and needs no network to verify — a **hand-crafted, correctly
signed webhook event was sent to the running server and correctly upgraded
a real user to Creator**, stored the fake customer/subscription IDs, and a
follow-up `subscription.deleted` event correctly downgraded them back to
free. A tampered signature was correctly rejected with 400.

## What could NOT be tested here (needs a real Stripe account + internet)

The one thing that could not be exercised is an actual **outbound** call
from this server to `api.stripe.com` — that hostname was unreachable from
this sandbox, same as pypi/npm/OpenAI before. Checkout-session creation and
the billing-portal call in `billing.py` are written against Stripe's
documented REST API (same endpoints their own SDKs wrap), but the very first
real test of "click Go Creator → land on an actual Stripe Checkout page" has
to happen once you deploy this somewhere with normal internet access and a
real Stripe account. If anything doesn't match (field names, response shape),
that's the first place to check.

## Setting up Stripe (~10 minutes, you do this part)

I don't create accounts or enter payment details on your behalf — here's
what you need to do once you're ready to accept real payments:

1. Create a Stripe account at stripe.com (free).
2. In the Dashboard, add a Product called e.g. "Klipp Creator" with a
   recurring price of €24/month. Copy its **Price ID** (`price_...`).
3. Get your **Secret key** (`sk_test_...` for testing, `sk_live_...` once
   you're ready for real charges) from Developers → API keys.
4. Set up a webhook endpoint pointing at
   `https://your-domain.com/api/billing/webhook`, subscribed to at least:
   `checkout.session.completed`, `customer.subscription.updated`,
   `customer.subscription.created`, `customer.subscription.deleted`. Copy the
   **Signing secret** (`whsec_...`).
5. Set these environment variables before starting the server:

```bash
export STRIPE_SECRET_KEY=sk_test_...
export STRIPE_WEBHOOK_SECRET=whsec_...
export STRIPE_PRICE_CREATOR=price_...
export APP_BASE_URL=https://your-domain.com   # used to build Stripe redirect URLs
export SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")  # signs session cookies — keep stable across restarts
```

Without `STRIPE_PRICE_CREATOR` set, the app runs fine — Free plan works
exactly as before, and the "Go Creator" button returns a clear 503 instead
of crashing.

## Explicitly out of scope for this pass

URL/link paste (YouTube etc. — deliberately left out, copyright/ToS
question worth deciding deliberately), team seats/approvals for Studio,
batch queue/API access, and the Sora/Kling/ElevenLabs integrations shown in
the landing page hero. Those are real product features, not small
additions — worth scoping separately.

## Running it

```bash
cd backend
pip install -r requirements.txt   # Flask/numpy/scipy/opencv are usually already
                                   # present on most systems; requests is used
                                   # for the OpenAI and Stripe REST calls
export OPENAI_API_KEY=sk-...      # optional — omit to run hook-detection fully offline
export STRIPE_SECRET_KEY=...      # optional — omit to run Free-plan only
export STRIPE_WEBHOOK_SECRET=...
export STRIPE_PRICE_CREATOR=...
export SECRET_KEY=...             # recommended — random per-restart otherwise, logging everyone out
python3 app.py                    # serves the app on http://localhost:8000
```

Then open `http://localhost:8000` — sign up, land on the pricing page, or
jump straight to `/app.html` to upload a video.

## Project layout

```
backend/
  app.py                Flask routes, login/quota enforcement, background job runner
  db.py                  SQLite: users, plans, rolling quota, contact requests
  auth.py                signup/login/logout/me, session auth guard
  billing.py             Stripe Checkout/webhook/portal + Studio contact form
  pipeline.py            orchestrates one job end-to-end (resolution tier aware)
  audio_analysis.py       offline signal extraction (energy/pace/flux/silence)
  highlight_scoring.py    sliding-window hook scoring + clip selection
  transcription.py        OpenAI Whisper + GPT ranking (optional, needs a key)
  captions.py             word-timestamps -> .ass subtitle file
  reframe.py              face detection + crop-rect math for each aspect
  render.py               ffmpeg command building; 1080p/4K aspect tables
frontend/
  index.html              landing page, pricing, sign up / sign in modal
  app.html                the clipping tool (requires login)
  account.html            plan, usage bar, upgrade / manage billing / sign out
jobs/                     per-upload working dirs + klipp.db (SQLite)
testdata/                 the synthetic video used to verify the pipeline
```

## Where to take this next

1. Deploy with real Stripe keys and click through an actual Checkout session
   end to end — this is the one path that was implemented against the spec
   but genuinely never run against Stripe's servers.
2. Same for `OPENAI_API_KEY` — confirm the captions/GPT-ranking path against
   the live API.
3. Swap Flask's dev server for a production one (gunicorn + a reverse proxy)
   and put the SQLite file on persistent storage (or move to Postgres if you
   expect real concurrent load).
4. Add password-reset (currently there's no "forgot password" flow) and
   email verification if you want it before going fully public.
5. If you want the URL-paste feature, `yt-dlp` is the standard tool — just
   be deliberate about which sources you allow.
