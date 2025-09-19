import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [uploaded, setUploaded] = useState(false);
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("https://chatbot-rag-react-js-8870a40a0709.herokuapp.com/upload_pdf/", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setUploaded(true);
        setMessages([{ role: "bot", text: data.message }]);
      } else {
        setMessages([{ role: "bot", text: data.detail || "Upload failed" }]);
      }
    } catch {
      setMessages([{ role: "bot", text: "Error: Unable to upload file." }]);
    }
  };

  const handleSend = async () => {
    if (!query.trim() || !uploaded) return;
    const userMessage = { role: "user", text: query };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setLoading(true);
    try {
      const response = await fetch("https://chatbot-rag-react-js-8870a40a0709.herokuapp.com/ask/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query }),
      });
      const data = await response.json();
      const botMessage = { role: "bot", text: data.answer || "No response" };
      setMessages((prev) => [...prev, botMessage]);
    } catch {
      const botMessage = { role: "bot", text: "Error: Unable to connect to backend." };
      setMessages((prev) => [...prev, botMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1 className="header">AI PDF Chatbot</h1>
      <div className="upload-section">
        <input type="file" accept="application/pdf" onChange={handleFileChange} />
        <button onClick={handleUpload}>Upload PDF</button>
      </div>
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.role === "user" ? "user" : "bot"}`}
          >
            {msg.text}
          </div>
        ))}
        {loading && <div className="loading">Thinking...</div>}
      </div>
      <div className="input-area">
        <input
          type="text"
          placeholder="Ask a question..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && handleSend()}
          disabled={!uploaded}
        />
        <button onClick={handleSend} disabled={!uploaded}>Send</button>
      </div>
    </div>
  );
}

export default App;
