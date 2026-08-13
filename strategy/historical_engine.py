"""
historical_engine.py — single-file historical market-movement strategy.

Drop-in contract:
    def build(ctx) -> HistoricalEngine

No existing bot module needs to be changed. The strategy:
  1) reads the existing live TradeStore/MarketDataStore;
  2) downloads public OKX historical trades internally;
  3) builds point-in-time features and future labels in memory;
  4) caches the historical feature rows;
  5) finds nearest historical market states;
  6) emits the existing Signal only when probability, expected return,
     match distance and sample count pass the configured gates.

Historical features never contain future information. Future return/MFE/MAE
are labels used only for historical scoring.

Note: OKX public historical-trade availability must be verified for the exact
instrument/period. The engine fails closed if it cannot build a usable dataset.
"""

from __future__ import annotations
import asyncio, csv, hashlib, json, logging, math, os, time, urllib.parse, urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from market_data import MarketDataStore, TradeStore, Signal, DEFAULT_SYMBOL_WHITELIST
from .base import StrategyContext, StrategyEngine

log = logging.getLogger("okx_futures.historical")
BASE = "https://www.okx.com"
HISTORY_PATH = "/api/v5/market/history-trades"

FEATURES = (
    "ret_10s","ret_30s","ret_1m","ret_5m","ret_15m",
    "accel_30s","accel_1m","buy_ratio","volume_ratio",
    "range_1m","range_5m","volatility_1m","volatility_5m",
)
WEIGHTS = {
    "ret_10s":1.0,"ret_30s":1.2,"ret_1m":1.4,"ret_5m":1.6,"ret_15m":1.2,
    "accel_30s":1.0,"accel_1m":1.1,"buy_ratio":1.2,"volume_ratio":1.0,
    "range_1m":0.8,"range_5m":0.8,"volatility_1m":0.8,"volatility_5m":0.8,
}

@dataclass
class HistoricalEngineConfig:
    history_days: int = 365
    history_cache_dir: str = ".historical_cache"
    history_refresh_hours: float = 24.0
    historical_sample_sec: int = 10
    historical_warmup_sec: int = 900

    live_window_ms: int = 900_000
    min_live_warmup_sec: float = 45.0
    min_live_trade_count: int = 20

    neighbors: int = 100
    min_neighbors: int = 30
    max_distance: float = 1.0
    allow_cross_symbol_matches: bool = True

    min_direction_probability: float = 0.65
    min_expected_return_pct: float = 0.05
    target_pct: float = 0.30
    stop_pct: float = 0.20

    cooldown_sec: float = 60.0
    max_observation_minutes: float = 6.0
    symbol_whitelist: Optional[frozenset] = field(
        default_factory=lambda: DEFAULT_SYMBOL_WHITELIST
    )

    request_timeout_sec: float = 20.0
    request_delay_sec: float = 0.12
    max_history_requests_per_refresh: int = 0
    require_requested_history: bool = False
    log_top_neighbors: int = 3

@dataclass(frozen=True)
class Trade:
    ts: float
    price: float
    qty: float
    side: str
    trade_id: str = ""

@dataclass
class Row:
    symbol: str
    ts: float
    x: Dict[str,float]
    r1: float
    r5: float
    r15: float
    mfe: float
    mae: float

@dataclass
class Candidate:
    symbol: str
    started_at: float = field(default_factory=time.time)
    status: str = "OBSERVING"
    last_checked_at: float = 0.0
    data_ready: bool = False
    direction: str = ""
    probability: float = 0.0
    expected_return_pct: float = 0.0
    neighbor_count: int = 0
    avg_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0
    quality_score: float = 0.0
    match_distance: float = float("inf")
    @property
    def elapsed_sec(self): return time.time() - self.started_at

def f(v, default=0.0):
    try:
        x=float(v); return x if math.isfinite(x) else default
    except (TypeError,ValueError): return default

def pct(a,b): return (b-a)/a if a>0 else 0.0
def mean(v): return sum(v)/len(v) if v else 0.0
def sd(v):
    if len(v)<2: return 0.0
    m=mean(v); return math.sqrt(sum((x-m)**2 for x in v)/len(v))

def timestamp(v):
    x=f(v,float("nan"))
    if math.isfinite(x): return x/1000 if x>10_000_000_000 else x
    try: return datetime.fromisoformat(str(v).replace("Z","+00:00")).timestamp()
    except Exception: return None

def side_of(item):
    s=str(item.get("side","")).lower()
    if s in ("buy","sell"): return s
    m=item.get("m")
    if isinstance(m,bool): return "sell" if m else "buy"
    if str(m).lower() in ("true","1"): return "sell"
    if str(m).lower() in ("false","0"): return "buy"
    try:
        w=int(item.get("way"))
        if 1<=w<=4: return "buy"
        if 5<=w<=8: return "sell"
    except Exception: pass
    return None

def okx_get(path, params, timeout):
    url=BASE+path+"?"+urllib.parse.urlencode(params)
    req=urllib.request.Request(url,headers={"User-Agent":"historical-engine/1.0","Accept":"application/json"})
    with urllib.request.urlopen(req,timeout=timeout) as r: data=json.loads(r.read().decode())
    if data.get("code")!="0": raise RuntimeError(f"OKX error {data.get('code')}: {data.get('msg')}")
    return data.get("data") or []

async def download(symbol,cfg):
    end=time.time(); start=end-cfg.history_days*86400
    cursor=str(int(end*1000)); rows=[]; requests=0
    while True:
        if cfg.max_history_requests_per_refresh and requests>=cfg.max_history_requests_per_refresh: break
        try:
            batch=await asyncio.to_thread(okx_get,HISTORY_PATH,{"instId":symbol,"limit":"100","before":cursor},cfg.request_timeout_sec)
        except Exception as e:
            log.error("[historical] %s download failed: %s",symbol,e); break
        requests+=1
        if not batch: break
        parsed=[]
        for z in batch:
            ts=timestamp(z.get("ts")); px=f(z.get("px")); qty=f(z.get("sz")); side=side_of(z)
            if ts is not None and px>0 and qty>0 and side:
                parsed.append(Trade(ts,px,qty,side,str(z.get("tradeId") or "")))
        if not parsed: break
        rows.extend(t for t in parsed if start<=t.ts<=end)
        oldest=min(t.ts for t in parsed)
        if oldest<=start: break
        nxt=str(int(oldest*1000)-1)
        if nxt==cursor: break
        cursor=nxt
        await asyncio.sleep(max(0,cfg.request_delay_sec))
    seen=set(); out=[]
    for t in sorted(rows,key=lambda x:(x.ts,x.trade_id)):
        k=(t.ts,t.price,t.qty,t.side,t.trade_id)
        if k not in seen: seen.add(k); out.append(t)
    return out

def win(ts, now, seconds):
    lo=now-seconds
    return [t for t in ts if lo<=t.ts<=now]

def features(ts,i):
    if i<1: return None
    now=ts[i].ts
    w={s:win(ts[:i+1],now,s) for s in (10,30,60,300,900)}
    def ret(s):
        a=w[s]
        return pct(a[0].price,a[-1].price) if len(a)>=2 else 0.0
    r10,r30,r1,r5,r15=(ret(s) for s in (10,30,60,300,900))
    w1,w5=w[60],w[300]
    p1=[t.price for t in w1]; p5=[t.price for t in w5]
    buy=sum(t.qty for t in w5 if t.side=="buy"); sell=sum(t.qty for t in w5 if t.side=="sell")
    total=buy+sell; v1=sum(t.qty for t in w1); v5=sum(t.qty for t in w5)
    def rv(a):
        return sd([pct(a[j-1].price,a[j].price) for j in range(1,len(a))]) if len(a)>=3 else 0.0
    return {
        "ret_10s":r10,"ret_30s":r30,"ret_1m":r1,"ret_5m":r5,"ret_15m":r15,
        "accel_30s":r10-r30/3,"accel_1m":r30/0.5-r1,
        "buy_ratio":buy/total if total else .5,
        "volume_ratio":v1/(v5/5) if v5 else 0.0,
        "range_1m":(max(p1)-min(p1))/p1[0] if p1 else 0.0,
        "range_5m":(max(p5)-min(p5))/p5[0] if p5 else 0.0,
        "volatility_1m":rv(w1),"volatility_5m":rv(w5),
    }

def outcomes(ts,i):
    entry=ts[i].price; now=ts[i].ts
    future=[t for t in ts[i+1:] if t.ts<=now+900]
    if not future: return None
    def close(s):
        q=[t for t in future if t.ts<=now+s]
        return pct(entry,q[-1].price) if q else 0.0
    p=[t for t in future if t.ts<=now+300]
    if not p: return None
    hi=max(t.price for t in p); lo=min(t.price for t in p)
    return close(60),close(300),close(900),max(0,(hi-entry)/entry),min(0,(lo-entry)/entry)

def build_rows(symbol,ts,cfg):
    if len(ts)<2:return []
    out=[]; next_sample=ts[0].ts+cfg.historical_warmup_sec; last=-1e99
    for i,t in enumerate(ts):
        if t.ts<next_sample or t.ts-last<cfg.historical_sample_sec: continue
        x=features(ts,i); y=outcomes(ts,i)
        if x is None or y is None: continue
        out.append(Row(symbol,t.ts,x,*y)); last=t.ts; next_sample=t.ts+cfg.historical_sample_sec
    return out

class Dataset:
    def __init__(self,rows):
        self.rows=rows
        self.mu={k:mean([r.x[k] for r in rows]) for k in FEATURES}
        self.sigma={k:max(sd([r.x[k] for r in rows]),1e-12) for k in FEATURES}
    def dist(self,a,r):
        num=sum(WEIGHTS[k]*(((a[k]-self.mu[k])/self.sigma[k]-(r.x[k]-self.mu[k])/self.sigma[k])**2) for k in FEATURES)
        den=sum(WEIGHTS.values())
        return math.sqrt(num/den)
    def nearest(self,x,symbol,now,cfg):
        q=[r for r in self.rows if r.ts<now]
        same=[r for r in q if r.symbol==symbol]
        if same:q=same
        elif not cfg.allow_cross_symbol_matches:q=[]
        z=sorted(((self.dist(x,r),r) for r in q),key=lambda a:a[0])
        return z[:max(1,cfg.neighbors)]

def label(row,direction,target,stop):
    if direction=="long": return row.mfe>=target/100 and abs(row.mae)<stop/100
    return abs(row.mae)>=stop/100 and row.mfe<target/100

def score(matches,direction,cfg):
    if not matches:return {"p":0,"e":0,"mfe":0,"mae":0,"q":0,"d":float("inf")}
    ret=[]; mfe=[]; mae=[]; wins=0
    for d,r in matches:
        ret.append(r.r5*(1 if direction=="long" else -1))
        if direction=="long": mfe.append(r.mfe); mae.append(abs(r.mae))
        else: mfe.append(abs(r.mae)); mae.append(r.mfe)
        wins+=label(r,direction,cfg.target_pct,cfg.stop_pct)
    n=len(matches); p=100*wins/n; e=100*mean(ret); mf=100*mean(mfe); ma=100*mean(mae)
    pc=max(0,min(100,p)); ec=max(0,min(100,50+e*50)); mc=max(0,min(100,(mf/max(.01,ma))*25))
    q=.50*pc+.30*ec+.20*mc
    return {"p":p,"e":e,"mfe":mf,"mae":ma,"q":q,"d":matches[0][0]}

class HistoricalEngine(StrategyEngine):
    name="historical_engine"
    def __init__(self,trade_store,market_data,candle_fetcher=None,config=None):
        self._trade_store=trade_store; self._market_data=market_data
        self._candle_fetcher=candle_fetcher; self.config=config or HistoricalEngineConfig()
        self._candidates={}; self._datasets={}; self._ready={}; self._tasks={}; self._last_signal={}
        self._lock=asyncio.Lock()

    def _cache(self,symbol):
        h=hashlib.sha1(symbol.encode()).hexdigest()[:10]
        return os.path.join(self.config.history_cache_dir,f"{symbol.replace('-','_')}_{h}.csv")

    async def sync_watchlist(self,symbols):
        symbols=set(symbols)
        if self.config.symbol_whitelist: symbols &= set(self.config.symbol_whitelist)
        async with self._lock:
            for s in symbols:
                self._candidates.setdefault(s,Candidate(s))
            for s in list(self._candidates):
                if s not in symbols:self._candidates.pop(s,None)
        for s in symbols:
            if s not in self._tasks or self._tasks[s].done():
                self._tasks[s]=asyncio.create_task(self._prepare(s))

    async def snapshot(self):
        async with self._lock:return list(self._candidates.values())

    async def _prepare(self,symbol):
        cfg=self.config; os.makedirs(cfg.history_cache_dir,exist_ok=True); path=self._cache(symbol)
        if os.path.exists(path) and time.time()-os.path.getmtime(path)<=cfg.history_refresh_hours*3600:
            try:
                rows=self._load(path,symbol)
                if rows:
                    self._datasets[symbol]=Dataset(rows); self._ready[symbol]=True
                    log.info("[historical] %s loaded %d cached states",symbol,len(rows)); return
            except Exception as e: log.warning("[historical] %s cache invalid: %s",symbol,e)
        trades=await download(symbol,cfg)
        if not trades:
            self._ready[symbol]=False; return
        covered=(trades[-1].ts-trades[0].ts)/86400
        if cfg.require_requested_history and covered<cfg.history_days*.9:
            log.error("[historical] %s only %.1f days available; refusing to trade",symbol,covered)
            self._ready[symbol]=False; return
        rows=build_rows(symbol,trades,cfg)
        if not rows:self._ready[symbol]=False; return
        self._write(path,rows); self._datasets[symbol]=Dataset(rows); self._ready[symbol]=True
        log.info("[historical] %s READY states=%d coverage=%.1f days",symbol,len(rows),covered)

    @staticmethod
    def _write(path,rows):
        fields=["symbol","timestamp",*FEATURES,"future_return_1m","future_return_5m","future_return_15m","mfe_5m","mae_5m"]
        tmp=path+".tmp"
        with open(tmp,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for r in rows:
                d={"symbol":r.symbol,"timestamp":r.ts,**r.x,"future_return_1m":r.r1,"future_return_5m":r.r5,"future_return_15m":r.r15,"mfe_5m":r.mfe,"mae_5m":r.mae}; w.writerow(d)
        os.replace(tmp,path)

    @staticmethod
    def _load(path,symbol):
        out=[]
        with open(path,newline="",encoding="utf-8") as f:
            for d in csv.DictReader(f):
                x={k:f(d.get(k)) for k in FEATURES}
                out.append(Row(d.get("symbol") or symbol,f(d.get("timestamp")),x,f(d.get("future_return_1m")),f(d.get("future_return_5m")),f(d.get("future_return_15m")),f(d.get("mfe_5m")),f(d.get("mae_5m"))))
        return out

    async def evaluate(self,symbol):
        cfg=self.config; c=self._candidates.get(symbol)
        if c is None:return None
        now=time.time(); c.last_checked_at=now
        if c.elapsed_sec>=cfg.max_observation_minutes*60:
            c.status="EXPIRED"
            async with self._lock:self._candidates.pop(symbol,None)
            return None
        if not self._ready.get(symbol):return None
        ds=self._datasets.get(symbol)
        if ds is None or now-self._last_signal.get(symbol,0)<cfg.cooldown_sec:return None
        market=await self._market_data.get(symbol)
        if not market:return None
        price=f(market.get("last_price"))
        if price<=0:return None
        trades=await self._trade_store.get_window(symbol,cfg.live_window_ms)
        c.data_ready=c.elapsed_sec>=cfg.min_live_warmup_sec and len(trades)>=cfg.min_live_trade_count
        if not c.data_ready:return None
        live=[]
        for t in trades:
            ts=f(t.get("timestamp")); ts=ts/1000 if ts>10_000_000_000 else ts
            if f(t.get("price"))>0 and f(t.get("qty"))>0:
                live.append(Trade(ts,f(t.get("price")),f(t.get("qty")),str(t.get("side"))))
        live.sort(key=lambda x:x.ts)
        x=features(live,len(live)-1) if len(live)>=2 else None
        if x is None:return None
        matches=ds.nearest(x,symbol,now,cfg)
        if len(matches)<cfg.min_neighbors:return None
        L=score(matches,"long",cfg); S=score(matches,"short",cfg)
        direction,st=("long",L) if L["e"]>S["e"] else ("short",S)
        c.direction=direction;c.probability=st["p"];c.expected_return_pct=st["e"];c.neighbor_count=len(matches);c.avg_mfe_pct=st["mfe"];c.avg_mae_pct=st["mae"];c.quality_score=st["q"];c.match_distance=st["d"]
        if st["p"]<cfg.min_direction_probability*100 or st["e"]<cfg.min_expected_return_pct or st["d"]>cfg.max_distance:return None
        log.info("[historical] ACCEPTED %s %s probability=%.1f%% expected5m=%+.3f%% MFE=%.3f%% MAE=%.3f%% neighbors=%d distance=%.3f quality=%.1f/100",symbol,direction.upper(),st["p"],st["e"],st["mfe"],st["mae"],len(matches),st["d"],st["q"])
        self._last_signal[symbol]=now
        async with self._lock:self._candidates.pop(symbol,None)
        return Signal(symbol=symbol,direction=direction,confidence=max(0,min(1,st["p"]/100)),entry_price=price,take_profit=price,stop_loss=price,timestamp=now,reasons=[
            "engine=historical_movement",f"historical_probability={st['p']:.1f}%",f"expected_return_5m={st['e']:+.3f}%",f"historical_mfe={st['mfe']:.3f}%",f"historical_mae={st['mae']:.3f}%",f"neighbors={len(matches)}",f"match_distance={st['d']:.3f}",f"quality_score={st['q']:.1f}"
        ])

def build(ctx: StrategyContext)->HistoricalEngine:
    cfg=ctx.build_config(HistoricalEngineConfig)
    return HistoricalEngine(ctx.trade_store,ctx.market_data,ctx.candle_fetcher,cfg)
