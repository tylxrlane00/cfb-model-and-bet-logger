/// <reference lib="deno.window" />
/// <reference lib="dom" />

// Weekly recap (no all-time leaderboard).
// Posts once per week (Sun 15:00 in TZ), guarded by bot_runs.
// Env: SB_URL, SB_SERVICE_ROLE_KEY, DISCORD_WEBHOOK_URL, ROOM, TZ

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { DateTime } from "https://esm.sh/luxon@3.4.4";

type BetRow = {
  timestamp: string;
  room: string;
  bettor: string | null;
  odds: number | null;
  stake: number | null;
  result: string | null;
};

const SB_URL = Deno.env.get("SB_URL")!;
const SB_SERVICE_ROLE_KEY = Deno.env.get("SB_SERVICE_ROLE_KEY")!;
const DISCORD_WEBHOOK_URL = Deno.env.get("DISCORD_WEBHOOK_URL")!;
const ROOM = Deno.env.get("ROOM") || "main";
const TZ = Deno.env.get("TZ") || "America/Chicago";

const sb = createClient(SB_URL, SB_SERVICE_ROLE_KEY);

function unitsFromAmerican(odds: number | null): number {
  if (odds === null || Number.isNaN(odds)) return 1.0;
  return odds < 0 ? 100 / Math.abs(odds) : odds / 100;
}
function profitUnits(r: BetRow): number {
  const stake = Number(r.stake ?? 1);
  const odds = r.odds ?? -110;
  const u = unitsFromAmerican(odds);
  const res = String(r.result || "pending").toLowerCase();
  if (res === "win") return stake * u;
  if (res === "loss") return -stake;
  return 0.0;
}

function weekStart(dt: DateTime): DateTime {
  // Luxon weeks start Monday
  return dt.startOf("week");
}

async function alreadySent(weekStartIsoDate: string): Promise<boolean> {
  const { data, error } = await sb
    .from("bot_runs")
    .select("id")
    .eq("room", ROOM)
    .eq("week_start", weekStartIsoDate)
    .limit(1);
  if (error) {
    console.log("bot_runs check error", error);
    return false;
  }
  return (data?.length || 0) > 0;
}
async function markSent(weekStartIsoDate: string): Promise<void> {
  const { error } = await sb.from("bot_runs").insert({
    room: ROOM,
    week_start: weekStartIsoDate,
  });
  if (error) console.log("bot_runs insert error", error);
}

function summarize(bets: BetRow[], bettor?: string) {
  const rows = bets
    .filter((b) => (b.result || "").toLowerCase() !== "deleted")
    .filter((b) => !bettor || b.bettor === bettor);
  const w = rows.filter((b) => (b.result || "").toLowerCase() === "win").length;
  const l = rows.filter((b) => (b.result || "").toLowerCase() === "loss").length;
  const p = rows.filter((b) => (b.result || "").toLowerCase() === "push").length;
  const units = rows.reduce((acc, r) => acc + profitUnits(r), 0);
  const risked = rows.reduce((acc, r) => acc + Number(r.stake ?? 0), 0);
  const roi = risked > 0 ? units / risked : 0;
  return { w, l, p, units, roi, n: rows.length };
}

function formatLine(name: string, s: ReturnType<typeof summarize>): string {
  return `${name.padEnd(12)}  ${`${s.w}-${s.l}-${s.p}`.padEnd(9)}  ${s.units >= 0 ? "+" : ""}${s.units.toFixed(2)}u  ROI ${(s.roi * 100).toFixed(1)}%  (${s.n} bets)`;
}

function buildMessage(weekTitle: string, weeklyBets: BetRow[]): string {
  const bettors = Array.from(
    new Set(weeklyBets.map((b) => b.bettor).filter(Boolean))
  ) as string[];

  const lines: string[] = [];
  lines.push(`**${weekTitle} — ${ROOM}**`);
  lines.push("```");
  if (bettors.length === 0) {
    lines.push("No graded bets for the period.");
  } else {
    for (const b of bettors) {
      const s = summarize(weeklyBets, b);
      lines.push(formatLine(b, s));
    }
    const tot = summarize(weeklyBets);
    lines.push("-".repeat(54));
    lines.push(formatLine("All bettors", tot));
  }
  lines.push("```");
  lines.push("────────────────────────────────");
  return lines.join("\n");
}

Deno.serve(async (req) => {
  try {
    const now = DateTime.now().setZone(TZ);
    const weekStartLocal = weekStart(now).startOf("day");
    const weekStartIsoDate = weekStartLocal.toISODate()!;

    // Only post on Sunday 15:00 local, unless forced
    const force = new URL(req.url).searchParams.get("force") === "1";
    const isSunday = now.weekday === 7;
    const is3pmTopOfHour = now.hour === 15 && now.minute === 0;

    if (!force) {
      if (!(isSunday && is3pmTopOfHour)) {
        return new Response(
          JSON.stringify({ ok: true, skipped: true, reason: "outside schedule", now: now.toISO() }),
          { headers: { "Content-Type": "application/json" } },
        );
      }
      if (await alreadySent(weekStartIsoDate)) {
        return new Response(
          JSON.stringify({ ok: true, skipped: true, reason: "already sent this week" }),
          { headers: { "Content-Type": "application/json" } },
        );
      }
    }

    // Weekly window: Monday 00:00 -> now (local)
    const startUTC = weekStartLocal.toUTC();
    const nowUTC = now.toUTC();

    const { data: weekly, error } = await sb
      .from("bet_logs")
      .select("timestamp, room, bettor, odds, stake, result")
      .eq("room", ROOM)
      .in("result", ["win", "loss", "push"])
      .gte("timestamp", startUTC.toISO()!)
      .lt("timestamp", nowUTC.toISO()!)
      .limit(5000);

    if (error) throw error;

    const title = `Weekly Recap ${weekStartLocal.toISODate()} → ${now.toISODate()}`;
    const content = buildMessage(title, (weekly || []) as BetRow[]);

    const resp = await fetch(DISCORD_WEBHOOK_URL, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ content }),
    });

    const ok = resp.ok;
    if (ok && !force) await markSent(weekStartIsoDate);

    return new Response(JSON.stringify({ ok, count_week: weekly?.length || 0 }), {
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
