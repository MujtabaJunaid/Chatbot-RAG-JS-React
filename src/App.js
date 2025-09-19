import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [ready, setReady] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loadingAnswer, setLoadingAnswer] = useState(false);

  const backendUrl = "https://chatbot-rag-react-js-8870a40a0709.herokuapp.com";

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setReady(false);
    const formData = new FormData();
    formData.append("file", file);
    await fetch(`${backendUrl}/upload_pdf/`, {
      method: "POST",
      body: formData,
    });
    setUploading(false);
  };

  useEffect(() => {
    let interval;
    if (uploading === false && file) {
      interval = setInterval(async () => {
        const res = await fetch(`${backendUrl}/status/`);
        const data = await res.json();
        if (data.ready) {
          setReady(true);
          clearInterval(interval);
        }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [uploading, file]);

  const handleAsk = async () => {
    if (!question) return;
    setLoadingAnswer(true);
    const res = await fetch(`${backendUrl}/ask/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    setAnswer(data.answer);
    setLoadingAnswer(false);
  };

  return (
    <div className="app">
      <h1>Income Tax Ordinance Chatbot</h1>
      <div className="upload-section">
        <input type="file" accept="application/pdf" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={uploading}>
          {uploading ? "Uploading..." : "Upload PDF"}
        </button>
      </div>
      {!ready && file && !uploading && <p>Processing PDF... Please wait</p>}
      {ready && (
        <div className="chat-section">
          <textarea
            placeholder="Ask a question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button onClick={handleAsk} disabled={loadingAnswer}>
            {loadingAnswer ? "Getting Answer..." : "Ask"}
          </button>
          {answer && (
            <div className="answer-box">
              <h3>Answer</h3>
              <p>{answer}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
