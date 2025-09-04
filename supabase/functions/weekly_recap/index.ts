// deno-lint-ignore-file no-explicit-any
// Deploy:  supabase functions deploy discord-bot --no-verify-jwt
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.4";

type Bet = {
  bettor: string;
  book?: string | null;
  type: "Spread" | "Total" | "Moneyline";
  side?: "Home" | "Away" | null;
  ou?: "Over" | "Under" | null;
  home_based_line?: number | null;
  total_line?: number | null;
  price?: number | null;
  stake?: number | null;
  status?: "pending" | "win" | "loss" | "push";
  note?: string | null;
  home_team: string;
  away_team: string;
};

function num(v: any, d = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

async function postDiscordJSON(webhook: string, body: any) {
  return fetch(webhook, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function notifyBet(webhook: string, payload: any) {
  // The app can send { op: "notify-bet", bet: {...} } OR just the bet object.
  const bet: Bet = payload.bet ?? payload;

  const price = num(bet.price);
  const stake = num(bet.stake, 1);

  let title = "";
  if (bet.type === "Total") {
    title = `${bet.bettor} bet: ${bet.ou} ${num(bet.total_line).toFixed(1)}`;
  } else if (bet.type === "Spread") {
    const who = bet.side === "Home" ? bet.home_team : bet.away_team;
    const ln = num(bet.home_based_line);
    title = `${bet.bettor} bet: ${bet.side} (${who}) ${ln > 0 ? `+${ln}` : ln}`;
  } else {
    const who = bet.side === "Home" ? bet.home_team : bet.away_team;
    title = `${bet.bettor} bet: ${bet.side} (${who}) ML`;
  }

  const lines: string[] = [
    `Matchup: **${bet.away_team} @ ${bet.home_team}**`,
    `Type: **${bet.type}**`,
  ];

  if (bet.type === "Total") {
    lines.push(`Selection: **${bet.ou} ${num(bet.total_line).toFixed(1)}**`);
  } else if (bet.type === "Spread") {
    const who = bet.side === "Home" ? bet.home_team : bet.away_team;
    const ln = num(bet.home_based_line);
    lines.push(`Selection: **${bet.side} (${who}) ${ln > 0 ? `+${ln}` : ln}**`);
  } else {
    const who = bet.side === "Home" ? bet.home_team : bet.away_team;
    lines.push(`Selection: **${bet.side} (${who}) ML**`);
  }

  lines.push(`Price: **${price}**`);
  lines.push(`Stake: **${stake}**`);
  if (bet.book) lines.push(`Book: **${bet.book}**`);
  if (bet.note) lines.push(`Note: ${bet.note}`);

  await postDiscordJSON(webhook, {
    embeds: [
      {
        title,
        description: lines.join("\n"),
        color: 0x1f8b4c,
        timestamp: new Date().toISOString(),
      },
    ],
  });

  return { ok: true };
}

async function weeklyRoundup(
  webhook: string,
  supabaseUrl: string,
  serviceRoleKey: string
) {
  const sb = createClient(supabaseUrl, serviceRoleKey);
  const since = new Date(Date.now() - 7 * 24 * 3600 * 1000).toISOString();

  const { data, error } = await sb
    .from("bets")
    .select("bettor,status,stake,price,created_at")
    .gte("created_at", since);

  if (error) {
    throw new Error(error.message);
  }

  const rec: Record<
    string,
    { win: number; loss: number; push: number; pending: number; count: number }
  > = {};
  for (const b of data ?? []) {
    const name = (b as any).bettor || "Unknown";
    rec[name] ??= { win: 0, loss: 0, push: 0, pending: 0, count: 0 };
    rec[name].count++;
    const st = (b as any).status ?? "pending";
    if (st in rec[name]) (rec[name] as any)[st]++;
    else rec[name].pending++;
  }

  const lines = Object.entries(rec).map(
    ([name, r]) => `**${name}** — ${r.win}-${r.loss}-${r.push}  (bets: ${r.count})`
  );

  const content =
    lines.length > 0
      ? `**Weekly Roundup** (last 7 days)\n${lines.join("\n")}`
      : "Weekly Roundup: No bets logged this week.";

  await postDiscordJSON(webhook, { content });
  return { ok: true, weekly: true };
}

export default async (req: Request) => {
  const WEBHOOK = Deno.env.get("DISCORD_WEBHOOK_URL");
  if (!WEBHOOK) return new Response("Missing DISCORD_WEBHOOK_URL", { status: 500 });

  try {
    // Supabase sets a schedule header for cron invocations.
    // We also allow manual HTTP with ?op=weekly-roundup
    const url = new URL(req.url);
    const opParam = url.searchParams.get("op");
    const scheduled =
      req.headers.get("x-scheduled") === "true" ||
      req.headers.get("x-supabase-schedule") === "true" ||
      opParam === "weekly-roundup";

    if (scheduled) {
      const SB_URL = Deno.env.get("SB_URL");
      const SR_KEY = Deno.env.get("SB_SERVICE_ROLE_KEY");
      if (!SB_URL || !SR_KEY) {
        return new Response(
          "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY",
          { status: 500 }
        );
      }
      const res = await weeklyRoundup(WEBHOOK, SB_URL, SR_KEY);
      return new Response(JSON.stringify(res), {
        headers: { "content-type": "application/json" },
      });
    }

    // Otherwise: treat as HTTP notify-bet call
    let payload: any = {};
    const ct = req.headers.get("content-type") || "";
    if (ct.includes("application/json")) payload = await req.json();

    // Accept both {op:"notify-bet", bet:{...}} and raw bet
    const res = await notifyBet(WEBHOOK, payload);
    return new Response(JSON.stringify(res), {
      headers: { "content-type": "application/json" },
    });
  } catch (e) {
    return new Response(`error: ${e}`, { status: 500 });
  }
};
