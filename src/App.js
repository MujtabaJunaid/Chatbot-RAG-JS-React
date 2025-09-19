import { useState } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const BACKEND = "https://chatbot-rag-react-js-8870a40a0709.herokuapp.com";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (!question || !question.trim()) {
      setError("Please enter a question");
      return;
    }
    setLoading(true);
    setAnswer("");
    try {
      const res = await fetch(`${BACKEND}/ask/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });
      const text = await res.text();
      if (!res.ok) {
        let detail = text;
        try {
          const parsed = JSON.parse(text);
          detail = parsed.detail || parsed.error || JSON.stringify(parsed);
        } catch {}
        throw new Error(detail || `HTTP ${res.status}`);
      }
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
      setAnswer(data?.answer || "No answer received");
    } catch (err) {
      setError(err.message || "Error contacting backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="card">
        <h1 className="title">Document Q&A</h1>
        <form onSubmit={handleSubmit}>
          <textarea
            className="input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Type your question here"
            rows={6}
          />
          <button className="btn" type="submit" disabled={loading}>
            {loading ? "Thinking..." : "Ask"}
          </button>
        </form>

        {error && <div className="error">{error}</div>}
        {answer && (
          <div className="answerBox">
            <h2 className="answerTitle">Answer</h2>
            <div className="answerText">{answer}</div>
          </div>
        )}
      </div>
      <footer className="footer">Backend: {BACKEND}</footer>
    </div>
  );
}

export default App;
