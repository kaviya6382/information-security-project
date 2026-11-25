
from flask import Flask, request, jsonify, render_template
import requests
import hashlib
import sqlite3
import os

app = Flask(__name__)

DATABASE = "stocks.db"

PROXY_URL = "https://stock-price-checker-proxy.freecodecamp.rocks/v1/stock"

def connect_db():
    return sqlite3.connect(DATABASE)

def initialize_db():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT PRIMARY KEY,
            likes INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_likes (
            symbol TEXT,
            ip_hash TEXT,
            UNIQUE(symbol, ip_hash)
        )
    """)

    conn.commit()
    conn.close()

def hash_ip(ip):
    return hashlib.sha256(ip.encode()).hexdigest()

def get_stock_price(symbol):
    url = f"{PROXY_URL}/{symbol}/quote"

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

    return {
        "stock": data.get("symbol", symbol),
        "price": float(data.get("latestPrice", data.get("price")))
    }

def add_like(symbol, ip_hash):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT OR IGNORE INTO stocks (symbol, likes) VALUES (?, 0)",
        (symbol,)
    )

    cursor.execute("""
        INSERT OR IGNORE INTO user_likes (symbol, ip_hash)
        VALUES (?, ?)
    """, (symbol, ip_hash))

    if cursor.rowcount == 1:
        cursor.execute("""
            UPDATE stocks
            SET likes = likes + 1
            WHERE symbol = ?
        """, (symbol,))

    conn.commit()
    conn.close()

def get_likes(symbol):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT likes FROM stocks WHERE symbol = ?",
        (symbol,)
    )

    result = cursor.fetchone()
    conn.close()

    return result[0] if result else 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/stock-prices", methods=["GET"])
def stock_prices():

    stocks = request.args.getlist("stock")
    like = request.args.get("like") == "true"

    if len(stocks) not in (1, 2):
        return jsonify({"error": "Provide one or two stocks"}), 400

    stocks = [s.strip().upper() for s in stocks]

    ip = request.remote_addr or "unknown"
    ip_hash = hash_ip(ip)

    results = []

    try:
        for symbol in stocks:

            if not symbol:
                return jsonify({"error": "Invalid stock symbol"}), 400

            if like:
                add_like(symbol, ip_hash)

            quote = get_stock_price(symbol)

            results.append({
                "stock": quote["stock"],
                "price": quote["price"],
                "likes": get_likes(symbol)
            })

        if len(results) == 1:
            return jsonify({"stockData": results[0]})

        difference = results[0]["likes"] - results[1]["likes"]

        stock_data = []

        for index, stock in enumerate(results):
            stock_data.append({
                "stock": stock["stock"],
                "price": stock["price"],
                "rel_likes": difference if index == 0 else -difference
            })

        return jsonify({"stockData": stock_data})

    except requests.RequestException:
        return jsonify({"error": "Unable to retrieve stock prices"}), 502

if __name__ == "__main__":
    initialize_db()
    app.run(debug=True, port=3000)
