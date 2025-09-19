import { useState } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!question) return;
    setLoading(true);
    setAnswer("");
    try {
      const res = await fetch("https://chatbot-rag-react-js-8870a40a0709.herokuapp.com/ask/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });
      const data = await res.json();
      setAnswer(data.answer || "No answer received");
    } catch {
      setAnswer("Error contacting backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Ask about the document</h1>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Type your question here"
      />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Loading..." : "Ask"}
      </button>
      {answer && <div className="answer">{answer}</div>}
    </div>
  );
}

export default App;
