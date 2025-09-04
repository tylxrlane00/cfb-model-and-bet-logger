# 🥜 The Goober Model — CFB Predictor

A Streamlit app for CFB matchup modeling (FPI + efficiencies + weather), market blending, EV, and a shared bet board backed by Supabase.  
New-bet alerts go to Discord immediately, and a weekly recap is posted via a Supabase Edge Function + (optional) `pg_cron`.

## Live App
<your Streamlit/Render URL here>

---

## One-Time Setup (Owner)

1. **Supabase schema**  
   Run the idempotent SQL in `supabase/migrations/0001_schema_and_cron.sql` (creates `bets`, `projections`, policies, etc.).

2. **Discord**  
   Create a webhook in your target channel; copy the **Webhook URL**.

3. **Edge Functions**
   - `discord-bot` — formats and posts **new bet** alerts (expects `DISCORD_WEBHOOK_URL`).
   - `weekly_recap` — posts the **weekly recap** (expects `SB_URL`, `SB_SERVICE_ROLE_KEY`, `DISCORD_WEBHOOK_URL`, `ROOM`, `TZ`).  
     Optionally schedule with `pg_cron` for Sundays (example: 21:00 UTC).

4. **Secrets / Env**
   Add the following to your hosting provider’s Secrets/Env panel (Streamlit: **Secrets**):

   ```toml
   # ---- App expects either these flat envs...
   SB_URL = "https://<project>.supabase.co"
   SB_SERVICE_KEY = "<service-role-or-service-key>"     # used for Edge calls
   DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/<id>/<token>"
   TZ = "America/Chicago"                               # weekly recap timezone

   # ---- ...or a supabase section (app reads both)
   [supabase]
   url = "https://<project>.supabase.co"
   anon_key = "<service-role-or service key for server-side>"
