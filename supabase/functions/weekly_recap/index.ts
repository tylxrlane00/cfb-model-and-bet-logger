/// <reference lib="deno.window" />
/// <reference lib="dom" />

// Weekly recap for public.bets (status in ['win','loss','push'])
// Env: SB_URL, SB_SERVICE_ROLE_KEY, DISCORD_WEBHOOK_URL, TZ (e.g. America/Chicago), TAG (optional grouping key)

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { DateTime } from "https://esm.sh/luxon@3.4.4";

type Bet = {
  created_at: string;
  bettor: string | null;
  price: number | null;
  stake: number | null;
  status: string | null;
};

const SB_URL = Deno.env.get("SB_URL")!;
const SB_SERVICE_ROLE_KEY = Deno.env.get("SB_SERVICE_ROLE_KEY")!;
const DISCORD_WEBHOOK_URL = Deno.env.get("DISCORD_WEBHOOK_URL")!;
const TZ = Deno.env.get("TZ") || "America/Chicago";
const TAG = Deno.env.get("TAG") || "default";

const sb = createClient(SB_URL, SB_SERVICE_ROLE_KEY);

// --- helpers ---
function unitsFromAmerican(odds: number | null): number {
  if (odds === null || Number.isNaN(odds)) return 1.0;
  return odds < 0 ? 100 / Math.abs(odds) : odds / 100;
}
function profitUnits(b: Bet): number {
  const stake = Number(b.stake ?? 1);
  const u = unitsFromAmerican(b.price ?? -110);
  const res = String(b.status || "pending").toLowerCase();
  if (res === "win") return stake * u;
  if (res === "loss") return -stake;
  return 0;
}
function weekStart(dt: DateTime): DateTime {
  return dt.startOf("week"); // Monday
}

async function alreadySent(weekStartIsoDate: string): Promise<boolean> {
  const { data, error } = await sb
    .from("bot_runs")
    .select("id")
    .eq("tag", TAG)
    .eq("week_start", weekStartIsoDate)
    .limit(1);
  if (error) {
    console.log("bot_runs check error", error);
    return false;
  }
  return (data?.length || 0) > 0;
}
async function markSent(weekStartIsoDate: string) {
  const { error } = await sb.from("bot_runs").insert({ tag: TAG, week_start: weekStartIsoDate });
  if (error) console.log("bot_runs insert error", error);
}

function summarize(bets: Bet[]) {
  const w = bets.filter(b => (b.status || "").toLowerCase() === "win").length;
  const l = bets.filter(b => (b.status || "").toLowerCase() === "loss").length;
  const p = bets.filter(b => (b.status || "").toLowerCase() === "push").length;
  const units = bets.reduce((acc, r) => acc + profitUnits(r), 0);
  const risked = bets.reduce((acc, r) => acc + Number(r.stake ?? 0), 0);
  const roi = risked > 0 ? units / risked : 0;
  return { w, l, p, units, roi, n: bets.length };
}

Deno.serve(async (req) => {
  try {
    const now = DateTime.now().setZone(TZ);
    const weekStartLocal = weekStart(now).startOf("day");
    const weekStartIsoDate = weekStartLocal.toISODate()!;

    const force = new URL(req.url).searchParams.get("force") === "1";
    const isSunday = now.weekday === 7;
    const is3pmTop = now.hour === 15 && now.minute === 0;

    if (!force) {
      if (!(isSunday && is3pmTop)) {
        return new Response(JSON.stringify({ ok: true, skipped: true, reason: "outside schedule" }), {
          headers: { "Content-Type": "application/json" },
        });
      }
      if (await alreadySent(weekStartIsoDate)) {
        return new Response(JSON.stringify({ ok: true, skipped: true, reason: "already sent" }), {
          headers: { "Content-Type": "application/json" },
        });
      }
    }

    // window: Monday 00:00 local -> now
    const startUTC = weekStartLocal.toUTC();
    const nowUTC = now.toUTC();

    const { data, error } = await sb
      .from("bets")
      .select("created_at,bettor,price,stake,status")
      .in("status", ["win", "loss", "push"])
      .gte("created_at", startUTC.toISO()!)
      .lt("created_at", nowUTC.toISO()!)
      .limit(5000);

    if (error) throw error;
    const rows = (data || []) as Bet[];

    // group by bettor
    const byBettor = new Map<string, Bet[]>();
    for (const r of rows) {
      const k = (r.bettor || "Unknown").toString();
      if (!byBettor.has(k)) byBettor.set(k, []);
      byBettor.get(k)!.push(r);
    }

    // embed
    const title = `Weekly Recap ${weekStartLocal.toISODate()} → ${now.toISODate()}`;
    const fields: Array<{ name: string; value: string; inline?: boolean }> = [];
    for (const [name, arr] of byBettor.entries()) {
      const s = summarize(arr);
      fields.push({
        name,
        value: `**${s.w}-${s.l}-${s.p}** • ${s.units >= 0 ? "+" : ""}${s.units.toFixed(2)}u • ROI ${(s.roi * 100).toFixed(1)}% • ${s.n} bets`,
        inline: false,
      });
    }
    const tot = summarize(rows);
    if (rows.length > 0) {
      fields.push({
        name: "All Bettors",
        value: `**${tot.w}-${tot.l}-${tot.p}** • ${tot.units >= 0 ? "+" : ""}${tot.units.toFixed(2)}u • ROI ${(tot.roi * 100).toFixed(1)}% • ${tot.n} bets`,
        inline: false,
      });
    }

    const embed = {
      title,
      description: rows.length ? "" : "No graded bets for the period.",
      color: 0x22c55e, // green
      timestamp: now.toUTC().toISO(),
      footer: { text: TAG },
      fields,
    };

    const resp = await fetch(DISCORD_WEBHOOK_URL, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ embeds: [embed] }),
    });

    const ok = resp.ok;
    if (ok && !force) await markSent(weekStartIsoDate);

    return new Response(JSON.stringify({ ok, count: rows.length }), {
      headers: { "Content-Type": "application/json" },
      status: ok ? 200 : 500,
    });
  } catch (e) {
    console.error(e);
    return new Response(JSON.stringify({ ok: false, error: String(e) }), {
      headers: { "Content-Type": "application/json" },
      status: 500,
    });
  }
});
