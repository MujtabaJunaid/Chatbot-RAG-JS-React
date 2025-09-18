import React, { useState } from "react";
import axios from "axios";
import { TextField, Button, Typography, Paper } from "@mui/material";

function Chatbot() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const res = await axios.post("https://your-heroku-app.herokuapp.com/ask/", {
        query: query,
      });
      setResponse(res.data.answer);
    } catch (error) {
      setResponse("Error: Unable to get response from server.");
    }
    setLoading(false);
  };

  return (
    <div style={{ display: "flex", justifyContent: "center", padding: "20px" }}>
      <Paper style={{ padding: "20px", width: "100%", maxWidth: "600px" }}>
        <Typography variant="h5" style={{ marginBottom: "20px" }}>
          Income Tax Ordinance Chatbot
        </Typography>
        <TextField
          label="Ask a question"
          variant="outlined"
          fullWidth
          value={query}
          onChange={handleQueryChange}
          style={{ marginBottom: "20px" }}
        />
        <Button
          variant="contained"
          color="primary"
          onClick={handleSubmit}
          fullWidth
          disabled={loading || !query}
        >
          {loading ? "Loading..." : "Submit"}
        </Button>
        {response && (
          <Typography variant="body1" style={{ marginTop: "20px" }}>
            {response}
          </Typography>
        )}
      </Paper>
    </div>
  );
}

export default Chatbot;
