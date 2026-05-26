import { useState, useCallback } from 'react'
import { api } from '../utils/api'

export function usePredict() {
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)

  const predict = useCallback(async (coinId, days = 7) => {
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
  }, [])

  return { predictions, loading, error, predict }
}

export function useMetrics() {
  const [metrics, setMetrics]   = useState(null)
  const [loading, setLoading]   = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.metrics()
      setMetrics(data)
    } catch (_) { /* ignore */ } finally {
      setLoading(false)
    }
  }, [])

  return { metrics, loading, refresh }
}
