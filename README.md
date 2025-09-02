# CFB Predictor – PPA Monte Carlo + Market Blend + EV

A Streamlit app for CFB matchup sims, market blend, EV, and a shared bet board backed by Supabase. Weekly recap posts to Discord via a Supabase Edge Function and `pg_cron`.

## Live App
<your Streamlit/Render URL here>

## One-time Setup (Owner)

1. **Supabase**: run `supabase/migrations/0001_schema_and_cron.sql` in SQL Editor.
2. **Discord**: create a webhook for your channel and copy its URL.
3. **Edge Function**: deployed (`weekly_recap`) and cron scheduled (Sunday 21:00 UTC).
4. **Secrets**: add Streamlit secrets (see below).

## Streamlit Secrets (server-side only)

Add in your hosting provider’s “Secrets/Env” panel:

```toml
SB_URL = "https://<project>.supabase.co"
SB_SERVICE_ROLE_KEY = "<service-role-key>"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/<id>/<token>"
ROOM = "room-name"
BETTORS = ["name1","name2","name3"]
