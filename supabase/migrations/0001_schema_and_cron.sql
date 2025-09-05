-- Tables (idempotent)
CREATE TABLE IF NOT EXISTS public.bets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  bettor text NOT NULL,
  book text,
  type text NOT NULL CHECK (type IN ('Spread','Total','Moneyline')),
  side text CHECK (side IN ('Home','Away')),
  ou text CHECK (ou IN ('Over','Under')),
  home_based_line numeric,
  total_line numeric,
  price numeric,
  stake numeric,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','win','loss','push')),
  note text,
  home_team text NOT NULL,
  away_team text NOT NULL
);

CREATE TABLE IF NOT EXISTS public.projections (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  home_team text NOT NULL,
  away_team text NOT NULL,
  home_pts numeric,
  away_pts numeric,
  total_model numeric,
  total_blended numeric,
  model_home_spread_str text,
  blended_home_spread_str text,
  recommendation text
);

-- Enable RLS (idempotent)
ALTER TABLE public.bets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.projections ENABLE ROW LEVEL SECURITY;

-- Create policies only if they don't already exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'bets' AND policyname = 'bets_select_all'
  ) THEN
    CREATE POLICY bets_select_all ON public.bets FOR SELECT USING (true);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'bets' AND policyname = 'bets_insert_all'
  ) THEN
    CREATE POLICY bets_insert_all ON public.bets FOR INSERT WITH CHECK (true);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'bets' AND policyname = 'bets_update_all'
  ) THEN
    CREATE POLICY bets_update_all ON public.bets FOR UPDATE USING (true) WITH CHECK (true);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'bets' AND policyname = 'bets_delete_all'
  ) THEN
    CREATE POLICY bets_delete_all ON public.bets FOR DELETE USING (true);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'projections' AND policyname = 'proj_select_all'
  ) THEN
    CREATE POLICY proj_select_all ON public.projections FOR SELECT USING (true);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'projections' AND policyname = 'proj_insert_all'
  ) THEN
    CREATE POLICY proj_insert_all ON public.projections FOR INSERT WITH CHECK (true);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = 'public' AND tablename = 'projections' AND policyname = 'proj_delete_all'
  ) THEN
    CREATE POLICY proj_delete_all ON public.projections FOR DELETE USING (true);
  END IF;
END $$;

-- New column (idempotent)
ALTER TABLE public.bets
  ADD COLUMN IF NOT EXISTS notify_discord boolean NOT NULL DEFAULT false;


create table if not exists public.bot_runs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  tag text not null,
  week_start date not null,
  unique (tag, week_start)
);
alter table public.bot_runs enable row level security;
do $$ begin
  if not exists (select 1 from pg_policies where schemaname='public' and tablename='bot_runs' and policyname='bot_runs_all') then
    create policy "bot_runs_all" on public.bot_runs for all using (true) with check (true);
  end if;
end $$;

ALTER TABLE public.projections
  ADD COLUMN IF NOT EXISTS market_home_spread_str text,
  ADD COLUMN IF NOT EXISTS p_cover_market_home numeric,
  ADD COLUMN IF NOT EXISTS sigma_margin_eff numeric,
  ADD COLUMN IF NOT EXISTS controls jsonb,
  ADD COLUMN IF NOT EXISTS saved_at timestamptz NOT NULL DEFAULT now();

-- Optional: backfill saved_at for existing rows that predate this column
UPDATE public.projections
SET saved_at = created_at
WHERE saved_at IS NULL;


-- ── Projections: add persisted market fields (idempotent)
ALTER TABLE public.projections
  ADD COLUMN IF NOT EXISTS market_spread_home numeric,
  ADD COLUMN IF NOT EXISTS market_total numeric,
  ADD COLUMN IF NOT EXISTS spread_price numeric,
  ADD COLUMN IF NOT EXISTS total_price numeric;

-- (You already added these earlier, but keeping here for completeness is harmless)
ALTER TABLE public.projections
  ADD COLUMN IF NOT EXISTS market_home_spread_str text,
  ADD COLUMN IF NOT EXISTS p_cover_market_home numeric,
  ADD COLUMN IF NOT EXISTS sigma_margin_eff numeric,
  ADD COLUMN IF NOT EXISTS controls jsonb,
  ADD COLUMN IF NOT EXISTS saved_at timestamptz NOT NULL DEFAULT now();

-- Optional backfill (safe to run repeatedly)
UPDATE public.projections
SET saved_at = created_at
WHERE saved_at IS NULL;

-- (Optional) Helpful index if you frequently sort by newest saved
CREATE INDEX IF NOT EXISTS idx_projections_saved_at ON public.projections(saved_at);
