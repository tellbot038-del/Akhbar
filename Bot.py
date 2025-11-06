#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram News-Bot | Translator: Gemini (اصلی) + DeepSeek (fallback)
نسخهٔ به‌روز شده: ذخیرهٔ کاربر اضافه‌کننده سایت و کاربر تاییدکننده خبر (added_by, approved_by)
"""
import os
import sqlite3
import asyncio
import logging
import html
import re
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI
import google.generativeai as genai
import httpx
from urllib.parse import urlparse, urlunparse

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Chat
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ---------- راه‌اندازی ----------
load_dotenv()
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)

BOT_TOKEN = os.getenv("BOT_TOKEN")
REVIEW_CH = os.getenv("REVIEW_CHANNEL_ID")  # مثلاً @reviewchnl یا عددی
PUBLISH_CH = os.getenv("PUBLISH_CHANNEL_ID")  # مثلاً @newschnl یا عددی

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "deepseek/deepseek-r1:free")

if not BOT_TOKEN:
    logging.critical("BOT_TOKEN is missing. Exiting.")
    raise SystemExit("BOT_TOKEN is required")

# Gemini client (در صورت موجود بودن)
if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        logging.warning("Failed to configure Gemini client: %s", e)
        gemini_model = None
else:
    gemini_model = None

# DeepSeek/OpenRouter (در صورت موجود بودن)
try:
    deepseek_client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1") if OPENROUTER_KEY else None
except Exception as e:
    logging.warning("Failed to configure OpenRouter/OpenAI client: %s", e)
    deepseek_client = None

# ---------- SQLite ----------
DB_NAME = "bot.db"


def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def init_db():
    with get_conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        # ایجاد جداول اولیه (اگر وجود نداشتند)
        conn.execute("CREATE TABLE IF NOT EXISTS sites(url TEXT PRIMARY KEY)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS news(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               url TEXT,
               title TEXT,
               translated TEXT,
               status TEXT DEFAULT 'pending',
               edited_by INTEGER,
               created DATETIME DEFAULT CURRENT_TIMESTAMP)"""
        )
        conn.commit()

        # اطمینان از ستون‌های جدید (مهاجرت ساده)
        def ensure_column(table: str, column: str, col_def: str):
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if column not in cols:
                logging.info("Altering table %s: adding column %s", table, column)
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
                conn.commit()

        # sites.added_by برای ثبت کاربری که سایت را اضافه کرده
        ensure_column("sites", "added_by", "INTEGER")

        # news.approved_by برای ثبت کاربری که خبر را تایید کرده
        ensure_column("news", "approved_by", "INTEGER")


# ---------- caches / pending ----------
REVIEW_ADMINS_CACHE = {"ids": set(), "ts": datetime.min}
REVIEW_ADMINS_TTL = timedelta(seconds=60)

PENDING_ADD = {}  # token -> {"url":..., "by": user_id, "ts": datetime}


# ---------- Utilities ----------
def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    parsed = urlparse(url)
    if not parsed.scheme:
        parsed = parsed._replace(scheme="http")
    if not parsed.netloc and parsed.path:
        p = parsed.path
        if re.match(r"^[\w\.-]+\.[a-zA-Z]{2,}", p):
            parsed = parsed._replace(netloc=p, path="")
    return urlunparse(parsed)


def escape_html(s: str) -> str:
    return html.escape(s or "")


async def run_db(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


# ---------- translation (async wrappers with retry) ----------
async def translate_gemini(text: str, tries: int = 2, delay: float = 0.5) -> str | None:
    if not gemini_model:
        return None
    for attempt in range(tries):
        try:
            def call():
                res = gemini_model.generate_content(f"Translate to Persian:\n{text}")
                if hasattr(res, "text") and res.text:
                    return res.text.strip()
                if isinstance(res, dict):
                    return (res.get("content", "") or res.get("text", "") or res.get("output", "")).strip()
                return None
            out = await asyncio.to_thread(call)
            if out:
                return out
        except Exception as e:
            logging.warning("Gemini attempt %d failed: %s", attempt + 1, e)
            await asyncio.sleep(delay * (attempt + 1))
    return None


async def translate_deepseek(text: str, tries: int = 2, delay: float = 0.5) -> str | None:
    if not deepseek_client:
        return None
    for attempt in range(tries):
        try:
            def call():
                try:
                    resp = deepseek_client.chat.completions.create(
                        model=FALLBACK_MODEL,
                        messages=[{"role": "user", "content": f"Translate to Persian:\n{text}"}],
                        temperature=0.2,
                        max_tokens=512,
                    )
                    if hasattr(resp, "choices") and resp.choices:
                        msg = resp.choices[0].message
                        if hasattr(msg, "content"):
                            return msg.content.strip()
                        if isinstance(msg, dict):
                            return msg.get("content", "").strip()
                    if isinstance(resp, dict):
                        ch = resp.get("choices")
                        if ch and isinstance(ch, list):
                            m = ch[0].get("message", {})
                            return m.get("content", "").strip()
                except Exception:
                    pass
                try:
                    resp = deepseek_client.chat.completions.create(
                        model=FALLBACK_MODEL,
                        messages=[{"role": "user", "content": f"Translate to Persian:\n{text}"}],
                    )
                    if hasattr(resp, "choices") and resp.choices:
                        return getattr(resp.choices[0].message, "content", None)
                except Exception as e:
                    logging.debug("DeepSeek inner fallback error: %s", e)
                return None
            out = await asyncio.to_thread(call)
            if out:
                return out
        except Exception as e:
            logging.warning("DeepSeek attempt %d failed: %s", attempt + 1, e)
            await asyncio.sleep(delay * (attempt + 1))
    return None


async def translate(text: str) -> str:
    res = await translate_gemini(text)
    if res:
        return res
    res = await translate_deepseek(text)
    if res:
        return res
    logging.warning("Both translators failed; returning original text")
    return text


# ---------- fetch titles (async) ----------
async def fetch_titles(url: str, count: int = 5, timeout: float = 10.0):
    url = normalize_url(url)
    titles = []
    try:
        async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}) as client:
            r = await client.get(url)
            r.raise_for_status()
            text = r.text
            soup = BeautifulSoup(text, "lxml")

            og = soup.find("meta", property="og:title")
            if og and og.get("content"):
                titles.append(og.get("content").strip())

            ttag = soup.find("title")
            if ttag and ttag.get_text(strip=True):
                titles.append(ttag.get_text(strip=True))

            for tag in ("h1", "h2"):
                for h in soup.find_all(tag):
                    txt = h.get_text(strip=True)
                    if txt and len(txt) > 10:
                        titles.append(txt)
                        if len(titles) >= count:
                            break
                if len(titles) >= count:
                    break

            if len(titles) < count:
                links = soup.find_all("a")
                for a in links:
                    txt = a.get_text(strip=True)
                    if txt and len(txt) > 20:
                        titles.append(txt)
                        if len(titles) >= count:
                            break

            seen = set()
            out = []
            for t in titles:
                if not t:
                    continue
                tt = re.sub(r"\s+", " ", t).strip()
                if tt.lower() in seen:
                    continue
                seen.add(tt.lower())
                out.append(tt)
                if len(out) >= count:
                    break
            return out
    except Exception as e:
        logging.error("fetch %s : %s", url, e)
        return []


# ---------- helpers for review-admin checking and message links ----------
async def refresh_review_admins_if_needed(bot) -> set:
    now = datetime.utcnow()
    if REVIEW_ADMINS_CACHE["ts"] + REVIEW_ADMINS_TTL > now and REVIEW_ADMINS_CACHE["ids"]:
        return REVIEW_ADMINS_CACHE["ids"]
    try:
        admins = await bot.get_chat_administrators(REVIEW_CH)
        ids = {m.user.id for m in admins}
        REVIEW_ADMINS_CACHE["ids"] = ids
        REVIEW_ADMINS_CACHE["ts"] = now
        logging.debug("Refreshed review admins: %s", ids)
        return ids
    except Exception as e:
        logging.warning("Could not fetch chat administrators for %s: %s", REVIEW_CH, e)
        return REVIEW_ADMINS_CACHE.get("ids", set())


async def user_is_review_admin(bot, user_id: int) -> bool:
    ids = await refresh_review_admins_if_needed(bot)
    return user_id in ids


async def make_message_link(bot, chat_id, message_id) -> str:
    try:
        chat = await bot.get_chat(chat_id)
        username = getattr(chat, "username", None)
        if username:
            return f"https://t.me/{username}/{message_id}"
        cid = int(chat.id)
        if cid < 0:
            s = str(cid)
            if s.startswith("-100"):
                s2 = s[4:]
            else:
                s2 = s.lstrip("-")
            return f"https://t.me/c/{s2}/{message_id}"
    except Exception as e:
        logging.debug("Failed to construct message link: %s", e)
    return ""


# ---------- send review ----------
async def send_review(app, nid: int, fa: str, link: str):
    safe_fa = escape_html(fa)
    safe_link = escape_html(normalize_url(link))
    kb = [[
        InlineKeyboardButton("✅ تایید", callback_data=f"apr_{nid}"),
        InlineKeyboardButton("📝 ویرایش", callback_data=f"edi_{nid}"),
    ]]
    text = f"📰 {safe_fa}\n\n🔗 <a href=\"{safe_link}\">منبع</a>"
    try:
        await app.bot.send_message(
            chat_id=REVIEW_CH,
            text=text,
            reply_markup=InlineKeyboardMarkup(kb),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except Exception as e:
        logging.error("Failed to send review message for nid=%s: %s", nid, e)


# ---------- cron ----------
async def cron(context: ContextTypes.DEFAULT_TYPE):
    logging.info("Cron started at %s", datetime.utcnow().isoformat())
    def read_sites():
        with get_conn() as conn:
            rows = [r[0] for r in conn.execute("SELECT url FROM sites").fetchall()]
        return rows
    sites = await run_db(read_sites)
    for url in sites:
        logging.info("Scraping site: %s", url)
        titles = await fetch_titles(url)
        logging.info("Found %d titles on %s", len(titles), url)
        for t in titles:
            def check_and_insert():
                with get_conn() as conn:
                    dup = conn.execute("SELECT 1 FROM news WHERE title=?", (t,)).fetchone()
                    if dup:
                        return None
                    cur = conn.execute(
                        "INSERT INTO news(url,title,translated) VALUES(?,?,?)",
                        (url, t, None),
                    )
                    nid = cur.lastrowid
                    conn.commit()
                    return nid
            nid = await run_db(check_and_insert)
            if not nid:
                continue
            fa = await translate(t)
            def update_translation():
                with get_conn() as conn:
                    conn.execute("UPDATE news SET translated=? WHERE id=?", (fa, nid))
                    conn.commit()
            await run_db(update_translation)
            await send_review(context.application, nid, fa, url)
    logging.info("Cron finished at %s", datetime.utcnow().isoformat())


# ---------- Handlers ----------
async def cmd_start(update: Update, _):
    await update.message.reply_text(
        "👋 ربات ترجمه‌ی اخبار آماده است.\n\n"
        "برای افزودن سایت لطفاً لینک سایت را در چت خصوصی برای من بفرستید یا از دستور زیر استفاده کنید:\n"
        "/add_site <URL>\n\n"
        "دستورات دیگر:\n/list_sites\n/del_site <URL>"
    )


async def cmd_add_site(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    allowed = await user_is_review_admin(context.application.bot, user_id)
    if not allowed:
        await update.message.reply_text("⛔ تنها ادمین‌های کانال بازبینی اجازهٔ افزودن سایت را دارند.")
        return
    try:
        url = update.message.text.split(maxsplit=1)[1].strip()
    except IndexError:
        await update.message.reply_text("❌ لطفاً آدرس را بعد دستور وارد کنید.")
        return
    norm = normalize_url(url)
    if not norm:
        await update.message.reply_text("❌ آدرس نامعتبر است.")
        return

    def do_insert():
        with get_conn() as conn:
            try:
                conn.execute("INSERT INTO sites(url, added_by) VALUES(?,?)", (norm, user_id))
                conn.commit()
                return True, None
            except sqlite3.IntegrityError:
                return False, "exists"
            except Exception as e:
                return False, str(e)

    ok, err = await run_db(do_insert)
    if ok:
        await update.message.reply_text("✅ اضافه شد.")
    else:
        if err == "exists":
            await update.message.reply_text("🔗 این سایت قبلاً ثبت شده.")
        else:
            await update.message.reply_text(f"❌ خطا در ثبت: {err}")


async def cmd_list(update: Update, _):
    def read():
        with get_conn() as conn:
            return [r[0] for r in conn.execute("SELECT url FROM sites").fetchall()]
    rows = await run_db(read)
    if not rows:
        await update.message.reply_text("📄 لیست خالی است.")
    else:
        await update.message.reply_text("\n".join(rows))


async def cmd_del(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    allowed = await user_is_review_admin(context.application.bot, user_id)
    if not allowed:
        await update.message.reply_text("⛔ تنها ادمین‌های کانال بازبینی اجازهٔ حذف سایت را دارند.")
        return
    try:
        url = update.message.text.split(maxsplit=1)[1].strip()
    except IndexError:
        await update.message.reply_text("❌ آدرس را وارد کنید.")
        return
    norm = normalize_url(url)
    def do_delete():
        with get_conn() as conn:
            c = conn.execute("DELETE FROM sites WHERE url=?", (norm,)).rowcount
            conn.commit()
            return c
    c = await run_db(do_delete)
    await update.message.reply_text("🗑️ حذف شد." if c else "پیدا نشد.")


# ---------- when user sends a URL in private chat -> propose add ----------
async def handle_private_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return
    if update.effective_chat.type != "private":
        return
    # find first http(s) url in text
    text = msg.text or ""
    m = re.search(r"(https?://[^\s]+)", text)
    if not m:
        await msg.reply_text("لینک پیدا نشد. لطفاً لینک کامل سایت را ارسال کنید.")
        return
    url = normalize_url(m.group(1))
    token = uuid.uuid4().hex
    PENDING_ADD[token] = {"url": url, "by": update.effective_user.id, "ts": datetime.utcnow()}
    kb = [
        [
            InlineKeyboardButton("➕ افزودن سایت", callback_data=f"addsite_{token}"),
            InlineKeyboardButton("❌ لغو", callback_data=f"addsite_cancel_{token}"),
        ]
    ]
    await msg.reply_text(f"آیا می‌خواهید این سایت را اضافه کنید؟\n{url}", reply_markup=InlineKeyboardMarkup(kb))


# ---------- callback handler (apr, edi, addsite, addsite_cancel) ----------
async def callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""
    user = update.effective_user.id

    # ------- addsite cancel -------
    if data.startswith("addsite_cancel_"):
        token = data.split("_", 2)[2] if "_" in data else None
        if token and token in PENDING_ADD:
            del PENDING_ADD[token]
        await q.message.edit_text("لغو شد.")
        return

    # ------- addsite confirm -------
    if data.startswith("addsite_"):
        token = data.split("_", 1)[1]
        info = PENDING_ADD.get(token)
        if not info:
            await q.message.reply_text("اطلاعات منقضی شده یا نامعتبر است.")
            return
        allowed = await user_is_review_admin(context.application.bot, user)
        if not allowed:
            await q.message.reply_text("⛔ تنها ادمین‌های کانال بازبینی اجازهٔ افزودن سایت را دارند.")
            return
        url = info["url"]
        # ثبت اضافه‌کننده به عنوان کاربرِ تاییدکننده (admin که افزود)
        def do_insert():
            with get_conn() as conn:
                try:
                    conn.execute("INSERT INTO sites(url, added_by) VALUES(?,?)", (url, user))
                    conn.commit()
                    return True, None
                except sqlite3.IntegrityError:
                    return False, "exists"
                except Exception as e:
                    return False, str(e)
        ok, err = await run_db(do_insert)
        if ok:
            await q.message.edit_text(f"✅ سایت اضافه شد:\n{url}")
        else:
            if err == "exists":
                await q.message.edit_text("🔗 این سایت قبلاً ثبت شده.")
            else:
                await q.message.edit_text(f"❌ خطا در ثبت: {err}")
        if token in PENDING_ADD:
            del PENDING_ADD[token]
        return

    # ------- approve / edit handling -------
    if data.startswith("apr_") or data.startswith("edi_"):
        parts = data.split("_", 1)
        if len(parts) != 2:
            await q.message.reply_text("دادهٔ نامعتبر.")
            return
        cmd, nid_s = parts
        try:
            nid = int(nid_s)
        except ValueError:
            await q.message.reply_text("شناسه نامعتبر.")
            return
        allowed = await user_is_review_admin(context.application.bot, user)
        if not allowed:
            await q.message.reply_text("⛔ دسترسی ندارید.")
            return

        if cmd == "apr":
            def do_approve():
                with get_conn() as conn:
                    row = conn.execute("SELECT translated,url FROM news WHERE id=?", (nid,)).fetchone()
                    if not row:
                        return None
                    fa, link = row
                    conn.execute("UPDATE news SET status='approved', approved_by=? WHERE id=?", (user, nid))
                    conn.commit()
                    return fa, link
            res = await run_db(do_approve)
            if not res:
                await q.message.reply_text("یافت نشد.")
                return
            fa, link = res
            safe_fa = escape_html(fa)
            safe_link = escape_html(normalize_url(link))
            text = f"📰 {safe_fa}\n\n🔗 <a href=\"{safe_link}\">منبع</a>"
            try:
                sent = await context.application.bot.send_message(
                    chat_id=PUBLISH_CH,
                    text=text,
                    parse_mode="HTML",
                    disable_web_page_preview=False,
                )
                # now construct link to published message and append it
                permalink = await make_message_link(context.application.bot, PUBLISH_CH, sent.message_id)
                if permalink:
                    new_text = text + f"\n\n🔗 لینک ترجمه‌شده: <a href=\"{escape_html(permalink)}\">مشاهده</a>"
                    try:
                        await context.application.bot.edit_message_text(
                            chat_id=PUBLISH_CH,
                            message_id=sent.message_id,
                            text=new_text,
                            parse_mode="HTML",
                            disable_web_page_preview=False,
                        )
                    except Exception as e:
                        logging.warning("Could not edit published message to add permalink: %s", e)
                await q.message.edit_text("✅ تایید و ارسال شد.")
            except Exception as e:
                logging.error("Failed to publish nid=%s: %s", nid, e)
                await q.message.reply_text("خطا در ارسال به کانال انتشار.")
        elif cmd == "edi":
            context.user_data["edit_nid"] = nid
            await q.message.reply_text("لطفاً ترجمه جدید را ارسال کنید.")
        else:
            await q.message.reply_text("دادهٔ نامعتبر.")
        return

    # default
    await q.message.reply_text("دادهٔ نامعتبر.")


# ---------- handle edited text ----------
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("edit_nid"):
        return
    nid = context.user_data.pop("edit_nid")
    new_text = update.message.text
    editor_id = update.effective_user.id

    def do_update():
        with get_conn() as conn:
            conn.execute("UPDATE news SET translated=?, status='edited', edited_by=? WHERE id=?", (new_text, editor_id, nid))
            row = conn.execute("SELECT url FROM news WHERE id=?", (nid,)).fetchone()
            conn.commit()
            return row[0] if row else None
    url = await run_db(do_update)
    await update.message.reply_text("✅ ویرایش ذخیره شد.")
    if url:
        await send_review(context.application, nid, new_text, url)


# ---------- main ----------
def main():
    init_db()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("add_site", cmd_add_site))
    app.add_handler(CommandHandler("list_sites", cmd_list))
    app.add_handler(CommandHandler("del_site", cmd_del))
    app.add_handler(CallbackQueryHandler(callback))  # handle many callback prefixes inside
    app.add_handler(MessageHandler(filters.Regex(r"https?://") & filters.ChatType.PRIVATE, handle_private_url))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.job_queue.run_repeating(cron, interval=30 * 60, first=10)
    logging.info("Bot started.")
    app.run_polling()


if __name__ == "__main__":
    main()
