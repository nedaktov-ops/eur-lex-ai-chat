import { useState, useRef, useEffect } from "react";

const API_URL = "https://nedaktovops-eurlex-chat-api.hf.space";

const CONFIDENCE_COLORS = {
  high: { bg: "bg-green-100", text: "text-green-800", label: "High confidence" },
  medium: { bg: "bg-yellow-100", text: "text-yellow-800", label: "Medium confidence" },
  low: { bg: "bg-red-100", text: "text-red-800", label: "Low confidence" },
};

function ConfidenceBadge({ level }) {
  const color = CONFIDENCE_COLORS[level] || CONFIDENCE_COLORS.low;
  if (!level) return null;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${color.bg} ${color.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${level === "high" ? "bg-green-500" : level === "medium" ? "bg-yellow-500" : "bg-red-500"}`} />
      {color.label}
    </span>
  );
}

function CitationLink({ celex }) {
  const url = `https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:${celex}`;
  return (
    <a href={url} target="_blank" rel="noopener noreferrer"
       className="text-blue-600 hover:text-blue-800 underline text-xs">
      {celex}
    </a>
  );
}

function SourceList({ sources }) {
  if (!sources || sources.length === 0) return null;
  // Deduplicate by CELEX
  const seen = new Set();
  const unique = sources.filter(s => {
    if (seen.has(s.celex)) return false;
    seen.add(s.celex);
    return true;
  });
  return (
    <div className="mt-3 pt-2 border-t border-gray-200">
      <p className="text-xs font-medium text-gray-500 mb-1">Sources:</p>
      <div className="flex flex-wrap gap-1.5">
        {unique.slice(0, 6).map((s) => (
          <span key={s.celex} className="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-100 rounded text-xs">
            <CitationLink celex={s.celex} />
            {s.article && <span className="text-gray-400">({s.article})</span>}
          </span>
        ))}
        {unique.length > 6 && (
          <span className="text-xs text-gray-400">+{unique.length - 6} more</span>
        )}
      </div>
    </div>
  );
}

function FeedbackButtons({ messageId, onFeedback }) {
  const [feedback, setFeedback] = useState(null);
  if (feedback) return null; // Already voted
  return (
    <div className="mt-1 flex gap-2 items-center">
      <button onClick={() => { setFeedback("up"); onFeedback?.(messageId, "up"); }}
              className="text-gray-400 hover:text-green-600 text-xs transition">
        👍 Helpful
      </button>
      <button onClick={() => { setFeedback("down"); onFeedback?.(messageId, "down"); }}
              className="text-gray-400 hover:text-red-600 text-xs transition">
        👎 Not helpful
      </button>
    </div>
  );
}

export default function ChatWidget() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi! I'm an AI assistant specialized in EU law. Ask me anything about EU regulations, directives, or legislation.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    const query = input.trim();
    if (!query || loading) return;

    setInput("");
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || `Error ${res.status}`);
      }

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          confidence: data._confidence,
          citations: data.citations || [],
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setError(err.message);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Sorry, I encountered an error: ${err.message}. Please try again.`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden max-w-2xl mx-auto">
      <div className="bg-[#003399] text-white px-4 py-3 font-semibold flex items-center justify-between">
        <span>Ask about EU Law</span>
        <span className="text-xs font-normal opacity-75">Powered by Groq Llama 3.3</span>
      </div>
      <div className="h-96 overflow-y-auto p-4 space-y-3 bg-gray-50">
        {messages.map((msg, i) => (
          <div key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[90%] rounded-lg px-4 py-2 text-sm whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-[#003399] text-white"
                  : "bg-white border border-gray-200 text-gray-800"
              }`}
            >
              {msg.confidence && (
                <div className="mb-1.5">
                  <ConfidenceBadge level={msg.confidence} />
                </div>
              )}
              <div className="leading-relaxed">{msg.content}</div>
              {msg.sources && msg.sources.length > 0 && (
                <SourceList sources={msg.sources} />
              )}
              {msg.role === "assistant" && i > 0 && (
                <FeedbackButtons
                  messageId={i}
                  onFeedback={(id, dir) => console.log("Feedback:", id, dir)}
                />
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-lg px-4 py-2 text-sm text-gray-500 italic">
              Thinking...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="border-t border-gray-200 p-3 flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about EU law..."
          rows={1}
          className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#003399] resize-none"
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="bg-[#003399] text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-[#002266] transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Send
        </button>
      </div>
      {error && (
        <div className="bg-red-50 text-red-700 text-xs px-4 py-2 border-t border-red-200">
          {error}
        </div>
      )}
    </div>
  );
}
