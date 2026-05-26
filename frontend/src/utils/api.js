const BASE = import.meta.env.VITE_API_URL || '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export const api = {
  predict:        (body)    => request('/predict',  { method: 'POST', body: JSON.stringify(body) }),
  feedback:       (body)    => request('/feedback', { method: 'POST', body: JSON.stringify(body) }),
  retrain:        (body)    => request('/retrain',  { method: 'POST', body: JSON.stringify(body) }),
  metrics:        ()        => request('/metrics'),
  feedbackHistory:()        => request('/feedback/history'),
  coins:          ()        => request('/coins'),
  health:         ()        => request('/health'),
}
