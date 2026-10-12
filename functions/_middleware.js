// 「券種ずれモニター」（zure-monitor.pages.dev）を本人だけが見られるようにする（Cloudflare Pages Functions）。
// - Pages のシークレット SITE_PASSWORD_SHA256（パスワードの SHA-256、16進64桁）と照合する。パスワードそのものは Cloudflare に置かない。
//   値は .github/workflows/site.yml が、リポジトリのシークレット SITE_PASSWORD から計算して登録する。
// - 合っていれば 90 日有効の Cookie を出し、以後はそれで通す。Cookie の値はハッシュから作るので、パスワードを変えると無効になる。
// - シークレットが未設定・形が違うときは、誰にも見せない（閉じたままにする）。
const COOKIE = "zm_session";
const MAX_AGE = 90 * 24 * 3600;
const PUBLIC = /^\/(manifest\.webmanifest|icon-\d+\.png)$/; // ホーム画面への追加に必要で、数字を含まないもの
const enc = new TextEncoder();
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");

async function sha256(s) {
  return hex(await crypto.subtle.digest("SHA-256", enc.encode(s)));
}

async function sessionToken(hash) {
  const key = await crypto.subtle.importKey("raw", enc.encode(hash), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  return hex(await crypto.subtle.sign("HMAC", key, enc.encode("zure-monitor-session-v1")));
}

// 長さ以外の違いで時間が変わらない比較
function same(a, b) {
  if (a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
}

const HEADERS = { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store", "X-Robots-Tag": "noindex", "Referrer-Policy": "no-referrer" };

function page(body, status) {
  return new Response(`<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>ログイン</title>
<style>body{font-family:system-ui,sans-serif;max-width:22rem;margin:18vh auto;padding:0 1rem;color:#1f3a5f}
input,button{font-size:1.1rem;padding:.6rem;width:100%;box-sizing:border-box;margin-top:.6rem}p{color:#b00020}</style></head>
<body>${body}</body></html>`, { status, headers: HEADERS });
}

const form = (msg) => `<h1>券種ずれモニター</h1>${msg ? `<p>${msg}</p>` : ""}
<form method="post"><input type="password" name="password" autocomplete="current-password" placeholder="パスワード" required autofocus>
<button type="submit">開く</button></form>`;

export async function onRequest({ request, env, next }) {
  const url = new URL(request.url);
  if (PUBLIC.test(url.pathname)) return next();
  const hash = String(env.SITE_PASSWORD_SHA256 || "").trim().toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(hash)) return page("<h1>準備中</h1><p>パスワードが未設定のため、閉じています。</p>", 503);
  const token = await sessionToken(hash);
  const cookies = (request.headers.get("Cookie") || "").split(/;\s*/);
  const mine = cookies.find((c) => c.startsWith(COOKIE + "="));
  if (mine && same(mine.slice(COOKIE.length + 1), token)) return next();
  if (request.method === "POST") {
    const data = await request.formData().catch(() => null);
    const pw = data ? String(data.get("password") || "") : "";
    if (pw && same(await sha256(pw), hash)) {
      return new Response(null, { status: 303, headers: { Location: "/", "Cache-Control": "no-store",
        "Set-Cookie": `${COOKIE}=${token}; Path=/; Max-Age=${MAX_AGE}; HttpOnly; Secure; SameSite=Lax` } });
    }
    await new Promise((r) => setTimeout(r, 1500)); // 総当たりを遅くする
    return page(form("パスワードが違います"), 401);
  }
  return page(form(""), 401);
}
