import React, { useState, useEffect, useRef } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, BarChart, Bar
} from 'recharts'
import {
  TrendingUp, TrendingDown, Activity, RefreshCw,
  ThumbsUp, ThumbsDown, Cpu, AlertTriangle, CheckCircle2,
  ChevronRight, Wifi, WifiOff, BarChart2
} from 'lucide-react'
import { api } from './utils/api'

// ─── CONSTANTS ───────────────────────────────────────────────────────────────

const COINS = [
  { id: 'bitcoin',  name: 'Bitcoin',  symbol: 'BTC', color: '#F7931A' },
  { id: 'ethereum', name: 'Ethereum', symbol: 'ETH', color: '#627EEA' },
  { id: 'cardano',  name: 'Cardano',  symbol: 'ADA', color: '#0033AD' },
  { id: 'solana',   name: 'Solana',   symbol: 'SOL', color: '#14F195' },
  { id: 'ripple',   name: 'Ripple',   symbol: 'XRP', color: '#00AAE4' },
]

const TABS = ['FORECAST', 'METRICS', 'HISTORY']

// ─── SMALL COMPONENTS ────────────────────────────────────────────────────────

function StatusDot({ online }) {
  return (
    <span className="flex items-center gap-1.5 text-xs font-display">
      <span className={`w-2 h-2 rounded-full ${online ? 'bg-accent animate-pulse' : 'bg-rose'}`} />
      <span className={online ? 'text-accent' : 'text-rose'}>
        {online ? 'API ONLINE' : 'API OFFLINE'}
      </span>
    </span>
  )
}

function Stat({ label, value, sub, accent }) {
  return (
    <div className="card flex flex-col gap-1">
      <span className="text-muted text-xs font-display tracking-widest uppercase">{label}</span>
      <span className={`text-2xl font-display font-bold ${accent ? 'text-accent glow-text' : 'text-text'}`}>
        {value}
      </span>
      {sub && <span className="text-muted text-xs">{sub}</span>}
    </div>
  )
}

function Loader() {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-24">
      <div className="w-12 h-12 border-2 border-border border-t-accent rounded-full animate-spin" />
      <span className="text-muted font-display text-sm tracking-widest">PROCESSING...</span>
    </div>
  )
}

function ErrorBanner({ msg, onDismiss }) {
  return (
    <div className="flex items-start gap-3 card border-rose/40 bg-rose/5 text-rose text-sm">
      <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
      <span className="flex-1">{msg}</span>
      <button onClick={onDismiss} className="opacity-60 hover:opacity-100 text-xs">✕</button>
    </div>
  )
}

// ─── CUSTOM TOOLTIP ──────────────────────────────────────────────────────────

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-surface border border-border rounded-lg px-3 py-2 text-xs font-display shadow-xl">
      <div className="text-muted mb-1">{label}</div>
      <div className="text-accent text-sm font-bold">
        ${Number(payload[0].value).toLocaleString(undefined, { maximumFractionDigits: 0 })}
      </div>
    </div>
  )
}

// ─── FORECAST TAB ────────────────────────────────────────────────────────────

function ForecastTab({ predictions, loading, error, onPredict, setError }) {
  const [coin, setCoin]             = useState('bitcoin')
  const [days, setDays]             = useState(7)
  const [feedbackSent, setFeedback] = useState(false)
  const [actualPrice, setActual]    = useState('')
  const [submitting, setSubmitting] = useState(false)

  const chartData = predictions
    ? [
        { date: 'Today', price: predictions.current_price, predicted: null },
        ...predictions.predictions.map(p => ({
          date: p.date.slice(5),
          price: null,
          predicted: p.price,
        })),
      ]
    : []

  const coinMeta = COINS.find(c => c.id === coin)
  const lastPred = predictions?.predictions?.at(-1)
  const firstPred = predictions?.predictions?.[0]
  const trend = firstPred ? (firstPred.change_percentage >= 0 ? 'up' : 'down') : null

  const handleFeedback = async (isAccurate) => {
    if (!predictions || !actualPrice) return
    setSubmitting(true)
    try {
      await api.feedback({
        prediction_id:  `${coin}_${Date.now()}`,
        actual_price:   parseFloat(actualPrice),
        predicted_price: predictions.predictions[0].price,
        date:           predictions.predictions[0].date,
        is_accurate:    isAccurate,
        user_comment:   isAccurate ? 'Accurate' : 'Needs improvement',
      })
      setFeedback(true)
      setTimeout(() => setFeedback(false), 4000)
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-5">
      {error && <ErrorBanner msg={error} onDismiss={() => setError(null)} />}

      {/* Controls */}
      <div className="card space-y-4">
        <h2 className="font-display text-xs tracking-widest text-muted uppercase">Select Asset</h2>
        <div className="flex flex-wrap gap-2">
          {COINS.map(c => (
            <button
              key={c.id}
              onClick={() => setCoin(c.id)}
              style={coin === c.id ? { borderColor: c.color, color: c.color } : {}}
              className={`px-4 py-2 rounded-lg border text-sm font-display transition-all duration-200
                ${coin === c.id
                  ? 'bg-white/5'
                  : 'border-border text-muted hover:border-accent/40 hover:text-text'
                }`}
            >
              {c.symbol}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-4">
          <div>
            <label className="text-muted text-xs font-display tracking-widest block mb-1.5">
              FORECAST WINDOW
            </label>
            <div className="flex gap-2">
              {[3, 7, 14].map(d => (
                <button
                  key={d}
                  onClick={() => setDays(d)}
                  className={`px-3 py-1.5 rounded-lg border text-sm font-display transition-all
                    ${days === d
                      ? 'border-accent text-accent bg-accent/10'
                      : 'border-border text-muted hover:border-accent/40'
                    }`}
                >
                  {d}D
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={() => onPredict(coin, days)}
            disabled={loading}
            className="ml-auto flex items-center gap-2 px-6 py-2.5 rounded-lg bg-accent text-bg
                       font-display font-bold text-sm tracking-wider hover:bg-accent/90
                       disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200
                       hover:shadow-lg hover:shadow-accent/20"
          >
            {loading ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
            RUN FORECAST
          </button>
        </div>
      </div>

      {/* Results */}
      {loading && <Loader />}

      {predictions && !loading && (
        <>
          {/* Stats row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat
              label="Current Price"
              value={`$${predictions.current_price.toLocaleString()}`}
              sub={coinMeta?.name}
              accent
            />
            <Stat
              label="Trend"
              value={trend === 'up' ? '▲ BULLISH' : '▼ BEARISH'}
              sub={`${firstPred?.change_percentage > 0 ? '+' : ''}${firstPred?.change_percentage?.toFixed(2)}% day 1`}
            />
            <Stat
              label="Confidence"
              value={`${(predictions.confidence * 100).toFixed(0)}%`}
              sub="volatility-adjusted"
            />
            <Stat
              label="Sentiment"
              value={predictions.sentiment.toUpperCase()}
              sub={`${predictions.sentiment_score}% bullish votes`}
            />
          </div>

          {/* Chart */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-display text-sm text-text tracking-wide">
                {coinMeta?.symbol} / USD — {days}-DAY FORECAST
              </h3>
              <span className="text-xs text-muted font-display">AI MODEL</span>
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <defs>
                  <linearGradient id="gActual" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#00FF88" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#00FF88" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gPred" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#FFB800" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#FFB800" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1C2433" />
                <XAxis dataKey="date" stroke="#4A5568" tick={{ fontSize: 11, fontFamily: 'Space Mono' }} />
                <YAxis stroke="#4A5568" tick={{ fontSize: 11, fontFamily: 'Space Mono' }}
                  tickFormatter={v => v ? `$${(v/1000).toFixed(0)}k` : ''} />
                <Tooltip content={<ChartTooltip />} />
                <ReferenceLine x="Today" stroke="#4A5568" strokeDasharray="4 4" />
                <Area type="monotone" dataKey="price" stroke="#00FF88" strokeWidth={2}
                  fill="url(#gActual)" dot={{ fill: '#00FF88', r: 4 }} connectNulls={false} />
                <Area type="monotone" dataKey="predicted" stroke="#FFB800" strokeWidth={2}
                  strokeDasharray="5 3" fill="url(#gPred)"
                  dot={{ fill: '#FFB800', r: 3 }} connectNulls={false} />
              </AreaChart>
            </ResponsiveContainer>
            <div className="flex gap-4 mt-3 text-xs font-display text-muted">
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-accent inline-block" /> ACTUAL</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-amber inline-block" /> PREDICTED</span>
            </div>
          </div>

          {/* Predictions table */}
          <div className="card">
            <h3 className="font-display text-sm text-text tracking-wide mb-4">DAILY BREAKDOWN</h3>
            <div className="space-y-2">
              {predictions.predictions.map(pred => (
                <div
                  key={pred.day}
                  className="flex items-center justify-between py-3 px-4
                             border border-border rounded-lg hover:border-accent/30
                             hover:bg-white/[0.02] transition-all duration-150"
                >
                  <div className="flex items-center gap-4">
                    <span className="font-display text-muted text-xs w-12">D+{pred.day}</span>
                    <span className="text-muted text-sm">{pred.date}</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="font-display font-bold text-text">
                      ${pred.price.toLocaleString()}
                    </span>
                    <span className={`flex items-center gap-1 text-sm font-display w-20 justify-end
                      ${pred.change_percentage >= 0 ? 'text-accent' : 'text-rose'}`}>
                      {pred.change_percentage >= 0
                        ? <TrendingUp className="w-3.5 h-3.5" />
                        : <TrendingDown className="w-3.5 h-3.5" />}
                      {pred.change_percentage >= 0 ? '+' : ''}{pred.change_percentage.toFixed(2)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Feedback */}
          <div className="card">
            <h3 className="font-display text-sm text-text tracking-wide mb-1">SUBMIT FEEDBACK</h3>
            <p className="text-muted text-sm mb-4">
              Enter the actual price and rate the prediction quality — this retrains the model over time.
            </p>

            {feedbackSent ? (
              <div className="flex items-center gap-2 text-accent text-sm font-display py-3">
                <CheckCircle2 className="w-4 h-4" />
                Feedback recorded. Thank you!
              </div>
            ) : (
              <div className="flex flex-col sm:flex-row gap-3">
                <div className="flex-1">
                  <label className="text-xs font-display text-muted tracking-widest block mb-1.5">
                    ACTUAL PRICE (USD)
                  </label>
                  <input
                    type="number"
                    placeholder={`e.g. ${predictions.current_price.toFixed(0)}`}
                    value={actualPrice}
                    onChange={e => setActual(e.target.value)}
                    className="w-full bg-bg border border-border rounded-lg px-4 py-2.5
                               text-text font-display text-sm focus:outline-none
                               focus:border-accent/60 transition-colors placeholder:text-muted/40"
                  />
                </div>
                <div className="flex gap-2 items-end">
                  <button
                    onClick={() => handleFeedback(true)}
                    disabled={!actualPrice || submitting}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-lg border border-accent/40
                               text-accent font-display text-sm hover:bg-accent/10 disabled:opacity-40
                               disabled:cursor-not-allowed transition-all"
                  >
                    <ThumbsUp className="w-4 h-4" /> Accurate
                  </button>
                  <button
                    onClick={() => handleFeedback(false)}
                    disabled={!actualPrice || submitting}
                    className="flex items-center gap-2 px-5 py-2.5 rounded-lg border border-rose/40
                               text-rose font-display text-sm hover:bg-rose/10 disabled:opacity-40
                               disabled:cursor-not-allowed transition-all"
                  >
                    <ThumbsDown className="w-4 h-4" /> Poor
                  </button>
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

// ─── METRICS TAB ─────────────────────────────────────────────────────────────

function MetricsTab() {
  const [metrics, setMetrics]     = useState(null)
  const [retraining, setRetrain]  = useState(false)
  const [coin, setCoin]           = useState('bitcoin')
  const [msg, setMsg]             = useState(null)

  useEffect(() => { loadMetrics() }, [])

  const loadMetrics = async () => {
    try { setMetrics(await api.metrics()) } catch (_) {}
  }

  const handleRetrain = async () => {
    setRetrain(true)
    setMsg(null)
    try {
      await api.retrain({ coin_id: coin, days: 730 })
      setMsg({ type: 'success', text: 'Retraining started in background (2–5 min).' })
      setTimeout(loadMetrics, 6000)
    } catch (e) {
      setMsg({ type: 'error', text: e.message })
    } finally {
      setRetrain(false)
    }
  }

  const barData = metrics
    ? [
        { name: 'Predictions', value: metrics.total_predictions },
        { name: 'Accurate',    value: metrics.accurate_predictions },
        { name: 'Feedback',    value: metrics.feedback_count },
      ]
    : []

  return (
    <div className="space-y-5">
      {msg && (
        <div className={`card border text-sm flex items-center gap-2
          ${msg.type === 'success'
            ? 'border-accent/40 bg-accent/5 text-accent'
            : 'border-rose/40 bg-rose/5 text-rose'
          }`}>
          {msg.type === 'success' ? <CheckCircle2 className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
          {msg.text}
        </div>
      )}

      {metrics ? (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <Stat label="Total Predictions" value={metrics.total_predictions} />
            <Stat label="Model Accuracy"    value={`${metrics.accuracy_percentage.toFixed(1)}%`} accent />
            <Stat label="Feedback Count"    value={metrics.feedback_count} />
            <Stat label="MAE"  value={`$${metrics.mae.toFixed(2)}`}  sub="Mean Absolute Error" />
            <Stat label="RMSE" value={`$${metrics.rmse.toFixed(2)}`} sub="Root Mean Square Error" />
            <Stat
              label="Last Retrain"
              value={metrics.last_retrain === 'Never'
                ? 'Never'
                : new Date(metrics.last_retrain).toLocaleDateString()
              }
              sub={metrics.last_retrain !== 'Never'
                ? new Date(metrics.last_retrain).toLocaleTimeString()
                : 'Run retrain to update'
              }
            />
          </div>

          <div className="card">
            <h3 className="font-display text-sm text-text tracking-wide mb-4">ACTIVITY OVERVIEW</h3>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={barData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1C2433" />
                <XAxis dataKey="name" stroke="#4A5568" tick={{ fontSize: 11, fontFamily: 'Space Mono' }} />
                <YAxis stroke="#4A5568" tick={{ fontSize: 11, fontFamily: 'Space Mono' }} />
                <Tooltip
                  contentStyle={{ background: '#0E1420', border: '1px solid #1C2433', borderRadius: 8, fontFamily: 'Space Mono', fontSize: 12 }}
                  cursor={{ fill: 'rgba(0,255,136,0.05)' }}
                />
                <Bar dataKey="value" fill="#00FF88" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h3 className="font-display text-sm text-text tracking-wide mb-3">MANUAL RETRAIN</h3>
            <p className="text-muted text-sm mb-4">
              Retrain automatically triggers after 50+ feedbacks. You can also force it here.
            </p>
            <div className="flex gap-3 flex-wrap">
              <select
                value={coin}
                onChange={e => setCoin(e.target.value)}
                className="bg-bg border border-border rounded-lg px-4 py-2.5 text-text
                           font-display text-sm focus:outline-none focus:border-accent/60"
              >
                {COINS.map(c => (
                  <option key={c.id} value={c.id}>{c.name}</option>
                ))}
              </select>
              <button
                onClick={handleRetrain}
                disabled={retraining}
                className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-accent text-bg
                           font-display font-bold text-sm hover:bg-accent/90 disabled:opacity-40
                           disabled:cursor-not-allowed transition-all"
              >
                <RefreshCw className={`w-4 h-4 ${retraining ? 'animate-spin' : ''}`} />
                {retraining ? 'STARTING...' : 'RETRAIN MODEL'}
              </button>
            </div>
          </div>
        </>
      ) : (
        <Loader />
      )}
    </div>
  )
}

// ─── HISTORY TAB ─────────────────────────────────────────────────────────────

function HistoryTab() {
  const [history, setHistory] = useState(null)

  useEffect(() => {
    api.feedbackHistory()
      .then(d => setHistory(d.feedback.slice().reverse()))
      .catch(() => setHistory([]))
  }, [])

  if (!history) return <Loader />

  if (history.length === 0) {
    return (
      <div className="card text-center py-16 text-muted font-display text-sm tracking-widest">
        NO FEEDBACK HISTORY YET
      </div>
    )
  }

  return (
    <div className="card overflow-hidden">
      <h3 className="font-display text-sm text-text tracking-wide mb-4">FEEDBACK LOG</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm font-display">
          <thead>
            <tr className="border-b border-border text-muted text-xs tracking-widest">
              <th className="text-left py-3 pr-4">DATE</th>
              <th className="text-right py-3 pr-4">PREDICTED</th>
              <th className="text-right py-3 pr-4">ACTUAL</th>
              <th className="text-right py-3 pr-4">ERROR</th>
              <th className="text-right py-3">STATUS</th>
            </tr>
          </thead>
          <tbody>
            {history.map((row, i) => (
              <tr key={i} className="border-b border-border/50 hover:bg-white/[0.02] transition-colors">
                <td className="py-3 pr-4 text-muted text-xs">{row.date}</td>
                <td className="py-3 pr-4 text-right text-text">
                  ${Number(row.predicted_price).toLocaleString()}
                </td>
                <td className="py-3 pr-4 text-right text-text">
                  ${Number(row.actual_price).toLocaleString()}
                </td>
                <td className="py-3 pr-4 text-right text-rose">
                  ${Number(row.error).toLocaleString()}
                </td>
                <td className="py-3 text-right">
                  <span className={`px-2 py-0.5 rounded text-xs
                    ${row.is_accurate
                      ? 'bg-accent/10 text-accent'
                      : 'bg-rose/10 text-rose'
                    }`}>
                    {row.is_accurate ? '✓ ACCURATE' : '✗ POOR'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─── ROOT APP ─────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab]             = useState('FORECAST')
  const [online, setOnline]       = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)

  // Health check on mount
  useEffect(() => {
    api.health()
      .then(() => setOnline(true))
      .catch(() => setOnline(false))
  }, [])

  const handlePredict = async (coinId, days) => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.predict({ coin_id: coinId, days })
      setPredictions(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="scanlines min-h-screen bg-bg text-text">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-surface/90 backdrop-blur-xl border-b border-border">
        <div className="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-accent/10 border border-accent/30
                            flex items-center justify-center">
              <TrendingUp className="w-4 h-4 text-accent" />
            </div>
            <span className="font-display font-bold text-sm tracking-widest">CRYPTOPREDICT</span>
          </div>

          <nav className="flex items-center gap-1">
            {TABS.map(t => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-1.5 rounded-lg text-xs font-display tracking-widest transition-all
                  ${tab === t
                    ? 'bg-accent/10 text-accent border border-accent/30'
                    : 'text-muted hover:text-text'
                  }`}
              >
                {t}
              </button>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            {online !== null && <StatusDot online={online} />}
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-5xl mx-auto px-4 py-8">
        {tab === 'FORECAST' && (
          <ForecastTab
            predictions={predictions}
            loading={loading}
            error={error}
            onPredict={handlePredict}
            setError={setError}
          />
        )}
        {tab === 'METRICS' && <MetricsTab />}
        {tab === 'HISTORY' && <HistoryTab />}
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-12 py-6 text-center">
        <p className="text-muted font-display text-xs tracking-widest">
          ⚠ EDUCATIONAL PROJECT — NOT FINANCIAL ADVICE &nbsp;|&nbsp; CRYPTOPREDICT © 2026
        </p>
      </footer>
    </div>
  )
}
