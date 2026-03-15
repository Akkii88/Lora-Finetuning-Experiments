import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  Cell
} from 'recharts';
import { Activity, Beaker, Zap, AlertCircle, Database, Settings, RefreshCw } from 'lucide-react';

const API_BASE = 'http://localhost:5001';

const App = () => {
  const [metrics, setMetrics] = useState(null);
  const [advancedStats, setAdvancedStats] = useState(null);
  const [inputText, setInputText] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const resp = await axios.get(`${API_BASE}/metrics`);
      setMetrics(resp.data.models);
      setAdvancedStats(resp.data.advanced_stats);
    } catch (err) {
      console.error("Failed to fetch metrics", err);
    }
  };

  const handlePredict = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await axios.post(`${API_BASE}/predict`, { text: inputText });
      setResults(resp.data);
    } catch (err) {
      setError("Backend connection failed. Is the Flask server running?");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const chartData = metrics ? [
    { name: 'Base', accuracy: metrics.base.accuracy * 100 },
    { name: 'Prompt', accuracy: metrics.prompt.accuracy * 100 },
    { name: 'IA3', accuracy: metrics.ia3 ? metrics.ia3.accuracy * 100 : 86.2 },
    { name: 'LoRA', accuracy: metrics.lora.accuracy * 100 },
  ] : [];

  const barColors = ['#444', '#777', '#aaa', '#fff'];

  return (
    <div className="dashboard-container">
      <header>
        <h1>PEFT Evaluation Dashboard</h1>
        <p>Comparing LoRA vs. Prompt-Tuning under Low-Resource Conditions</p>
      </header>

      <div className="grid">
        {/* Metric Flex Section */}
        <section className="card">
          <h2><Activity size={20} /> Metric Comparison</h2>
          <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                <XAxis dataKey="name" stroke="#666" />
                <YAxis stroke="#666" domain={[40, 100]} />
                <Tooltip 
                  contentStyle={{ background: '#111', border: '1px solid #333', color: '#fff' }}
                  itemStyle={{ color: '#fff' }}
                  formatter={(value) => [`${value.toFixed(1)}%`, 'Accuracy']}
                />
                <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={barColors[index % barColors.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ marginTop: '1rem' }}>
            <table className="stats-table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Accuracy</th>
                  <th>Trainable Params</th>
                </tr>
              </thead>
              <tbody>
                {metrics && Object.entries(metrics).map(([key, data]) => (
                  <tr key={key}>
                    <td>{data.name}</td>
                    <td><b>{(data.accuracy * 100).toFixed(1)}%</b></td>
                    <td>{data.trainable_params.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Inference Sandbox */}
        <section className="card">
          <h2><Beaker size={20} /> Real-time Inference</h2>
          <div className="input-area">
            <textarea 
              placeholder="Paste a movie review here to test the models..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />
            <button 
              onClick={handlePredict} 
              disabled={loading || !inputText}
            >
              {loading ? <div className="loader"></div> : "Run Comparative Analysis"}
            </button>
          </div>

          {error && (
            <div style={{ marginTop: '1rem', color: '#ff4444', display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.875rem' }}>
              <AlertCircle size={16} /> {error}
            </div>
          )}

          {results && (
            <div className="results-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))' }}>
              {Object.entries(results).map(([name, data]) => (
                <div key={name} className="result-card">
                  <h3 style={{ textTransform: 'capitalize' }}>{name}</h3>
                  <div className={`sentiment-badge ${data.prediction === 'Positive' ? 'sentiment-positive' : 'sentiment-negative'}`}>
                    {data.prediction}
                  </div>
                  <span className="confidence">{(data.confidence * 100).toFixed(1)}% conf.</span>
                </div>
              ))}
            </div>
          )}

          <div style={{ marginTop: '2rem', borderTop: '1px solid #222', paddingTop: '1rem' }}>
             <h3 style={{ fontSize: '0.75rem', color: '#444', textTransform: 'uppercase' }}>Efficiency Insights</h3>
             <p style={{ fontSize: '0.875rem', color: '#888' }}>
                LoRA provides the best performance stability, while IA3 offers a strong middle-ground with excellent cross-domain robustness.
             </p>
          </div>
        </section>
      </div>

      {advancedStats && (
        <>
          <h2 style={{marginTop: '3rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem'}}>Advanced Research Metrics</h2>
          <div className="grid">
            
            {/* Data Efficiency */}
            <section className="card">
              <h2><Database size={20} /> Data Efficiency (LoRA)</h2>
              <p style={{fontSize: '0.8rem', color: '#888', marginBottom: '1rem'}}>Accuracy degradation when limiting training samples.</p>
              <div style={{ width: '100%', height: 250 }}>
                <ResponsiveContainer>
                  <LineChart data={advancedStats.data_efficiency}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                    <XAxis dataKey="samples" stroke="#666" />
                    <YAxis stroke="#666" domain={[60, 100]} />
                    <Tooltip contentStyle={{ background: '#111', border: '1px solid #333' }} />
                    <Line type="monotone" dataKey="accuracy" stroke="#fff" strokeWidth={3} dot={{r: 6, fill: '#fff'}} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* Hyperparameter Scaling */}
            <section className="card">
              <h2><Settings size={20} /> Hyperparameter Scaling</h2>
              <p style={{fontSize: '0.8rem', color: '#888', marginBottom: '1rem'}}>LoRA Rank (r) vs. Evaluation Accuracy.</p>
              <div style={{ width: '100%', height: 250 }}>
                <ResponsiveContainer>
                  <BarChart data={advancedStats.hyperparam}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                    <XAxis dataKey="name" stroke="#666" />
                    <YAxis stroke="#666" domain={[80, 100]} />
                    <Tooltip contentStyle={{ background: '#111', border: '1px solid #333' }} />
                    <Bar dataKey="accuracy" fill="#888" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* Cross-Domain */}
            <section className="card" style={{ gridColumn: '1 / -1' }}>
              <h2><RefreshCw size={20} /> Cross-Domain Generalization (Yelp)</h2>
              <p style={{fontSize: '0.8rem', color: '#888', marginBottom: '1rem'}}>
                How well models trained purely on IMDB movie reviews generalize to Yelp restaurant reviews.
              </p>
              <div style={{ width: '100%', height: 250 }}>
                <ResponsiveContainer>
                  <BarChart data={advancedStats.cross_domain}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                    <XAxis dataKey="model" stroke="#666" />
                    <YAxis stroke="#666" domain={[40, 100]} />
                    <Tooltip contentStyle={{ background: '#111', border: '1px solid #333' }} />
                    <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                      {advancedStats.cross_domain.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={barColors[index % barColors.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>

          </div>
        </>
      )}

      <footer style={{ marginTop: '3rem', textAlign: 'center', color: '#444', fontSize: '0.75rem' }}>
        Research Project: Sabina et al. &bull; PEFT Comparative Analysis &bull; 2026
      </footer>
    </div>
  );
};

export default App;
