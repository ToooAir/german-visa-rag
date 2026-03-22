import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';
import { MoreVertical, Paperclip, Mic, Send, Bot, User, ShieldCheck, ExternalLink, Loader2, Sparkles, Database, FileSearch, Sparkles as SparklesIcon } from 'lucide-react';
import { useChatStore, Message, Source } from '../stores/chatStore';
import { useTranslation } from '../translations';

export function ChatArea({ className = '' }: { className?: string }) {
  const { messages, isLoading, sendMessage } = useChatStore();
  const [input, setInput] = useState('');
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const { t } = useTranslation();

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTo({
        top: scrollContainerRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [messages]);

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      sendMessage(input.trim());
      setInput('');
    }
  };

  return (
    <div className={`flex flex-col h-full bg-white/40 dark:bg-slate-900/40 rounded-2xl relative border border-slate-200/50 dark:border-slate-800/50 shadow-2xl overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-200/50 dark:border-slate-800/50 bg-white/50 dark:bg-slate-900/50 backdrop-blur-md z-10">
        <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100 flex items-center gap-2">
          {t.chatHeader} {isLoading && <Loader2 className="w-4 h-4 animate-spin text-accent" />}
        </h2>
        <div className="flex items-center gap-3 text-slate-500 dark:text-slate-400">
          <button className="p-1 hover:text-slate-900 dark:hover:text-white transition-colors relative">
            <User size={20} />
            <div className="absolute top-0 right-0 w-2 h-2 bg-green-500 rounded-full border border-white dark:border-slate-900" />
          </button>
          <button className="p-1 hover:text-slate-900 dark:hover:text-white transition-colors bg-slate-200 dark:bg-slate-800 rounded-full w-8 h-8 flex items-center justify-center text-xs font-bold font-mono">
            U
          </button>
        </div>
      </div>

      {/* Message List */}
      <div 
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto p-4 flex flex-col gap-6 pb-24"
      >
        <AnimatePresence>
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}
        </AnimatePresence>
        {messages.length === 1 && !isLoading && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <QuickStarters onSelect={sendMessage} />
          </motion.div>
        )}
      </div>

      {/* Input Area */}
      <div className="absolute bottom-0 w-full bg-gradient-to-t from-background via-background to-transparent pb-6 pt-12 z-20">
        <div className="relative glass-panel bg-white/60 dark:bg-slate-800/60 backdrop-blur-xl border border-slate-200 dark:border-slate-700 mx-4 flex items-center shadow-2xl shadow-black/5 dark:shadow-black/50 overflow-hidden">
          <textarea 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder={t.askPlaceholder} 
            className="w-full bg-transparent text-slate-800 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 py-4 pl-4 pr-32 outline-none text-[15px] resize-none overflow-hidden"
            rows={1}
            style={{ minHeight: '56px' }}
          />
          <div className="absolute right-3 flex items-center gap-1">
            <button className="p-2 text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors rounded-lg hover:bg-slate-100/50 dark:hover:bg-slate-700/50" disabled={isLoading}>
              <Paperclip size={18} />
            </button>
            <button className="p-2 text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors rounded-lg hover:bg-slate-100/50 dark:hover:bg-slate-700/50" disabled={isLoading}>
              <Mic size={18} />
            </button>
            <button 
              onClick={handleSend}
              disabled={isLoading || !input.trim()}
              className="p-2 ml-1 text-white bg-accent hover:bg-accent/80 transition-colors rounded-xl shadow-lg shadow-accent/20 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
        
        {/* Subtle Disclaimer */}
        <div className="flex justify-center px-4 mt-3">
          <p className="text-[10px] text-slate-400 dark:text-slate-500 text-center leading-tight max-w-2xl opacity-60">
            {t.disclaimer}
          </p>
        </div>
      </div>
    </div>
  );
}

function ChatMessage({ message }: { message: Message }) {
  const isUser = message.role === 'user';
  const { isLoading } = useChatStore();
  const { t } = useTranslation();
  
  if (isUser) {
    return (
      <motion.div 
        initial={{ opacity: 0, x: 20 }} 
        animate={{ opacity: 1, x: 0 }}
        className="glass-panel p-4 flex flex-col gap-2 max-w-[85%] self-end bg-slate-50/80 dark:bg-slate-800/40"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center text-xs font-bold text-slate-600 dark:text-slate-300">
              U
            </div>
            <span className="font-medium text-slate-800 dark:text-slate-200 text-sm">{t.you}</span>
          </div>
          <div className="flex items-center gap-2 text-slate-400 dark:text-slate-500 text-xs">
            <MoreVertical size={14} className="cursor-pointer hover:text-slate-600 dark:hover:text-slate-300" />
          </div>
        </div>
        <div className="text-slate-700 dark:text-slate-300 text-[15px] leading-relaxed ml-11 whitespace-pre-wrap">
          {message.content}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="glass-panel p-5 flex flex-col gap-3 w-full bg-white dark:bg-panel border-l-2 border-l-accent shadow-lg relative"
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center text-accent ring-1 ring-accent/30">
            <Bot size={18} />
          </div>
          <span className="font-semibold text-slate-900 dark:text-white text-sm tracking-wide">VisaPilot AI</span>
        </div>
      </div>
      
      <div className="text-slate-700 dark:text-slate-300 text-[15px] leading-relaxed ml-11 flex flex-col gap-4">
        {/* RAG Traceability Loader */}
        {isLoading && !message.content ? (
          <TraceLoader currentStatus={message.status} />
        ) : (
          <div className="prose prose-slate dark:prose-invert prose-p:leading-relaxed prose-pre:bg-slate-100 dark:prose-pre:bg-slate-800/50 prose-pre:border prose-pre:border-slate-200 dark:prose-pre:border-slate-700 max-w-none prose-a:text-accent hover:prose-a:text-accent/80 prose-a:font-medium prose-a:no-underline hover:prose-a:underline prose-strong:text-slate-800 dark:prose-strong:text-slate-200">
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                a: ({ ...props }) => (
                  <a {...props} target="_blank" rel="noopener noreferrer" className="text-accent underline decoration-accent/30 underline-offset-4 hover:decoration-accent transition-all" />
                )
              }}
            >
              {message.id === 'welcome' ? t.welcome : (message.content || '...')}
            </ReactMarkdown>
          </div>
        )}

        {/* Source Chips */}
        {message.sources && message.sources.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2 pt-3 border-t border-slate-200/50 dark:border-slate-700/50">
            {Array.from(new Map(message.sources.map(src => [src.url, src])).values()).map((src, i) => (
              <SourceChip key={i} source={src} />
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

function SourceChip({ source }: { source: Source }) {
  const isOfficial = source.authority === 'official';
  const isSemi = source.authority === 'semi_official';
  
  return (
    <motion.a 
      whileHover={{ y: -1 }}
      href={source.url} 
      target="_blank" 
      rel="noopener noreferrer"
      className="group inline-flex items-center gap-1.5 px-3 py-1.5 bg-slate-50/80 dark:bg-slate-800/40 hover:bg-slate-100 dark:hover:bg-slate-700/50 border border-slate-200/80 dark:border-slate-700/80 rounded-lg transition-all duration-200 hover:shadow-sm cursor-pointer max-w-[240px]"
    >
      {isOfficial ? (
        <ShieldCheck size={13} className="text-blue-500 dark:text-blue-400 shrink-0" />
      ) : isSemi ? (
        <ShieldCheck size={13} className="text-slate-400 shrink-0" />
      ) : (
        <ExternalLink size={13} className="text-slate-400 shrink-0" />
      )}
      <span className="text-xs font-medium text-slate-700 dark:text-slate-300 truncate leading-none pt-[1px]">
        {source.title || new URL(source.url).hostname.replace('www.', '')}
      </span>
    </motion.a>
  );
}

function QuickStarters({ onSelect }: { onSelect: (q: string) => void }) {
  const { t } = useTranslation();
  const starters = [
    { title: t.showcase.pointCalc.title, query: t.showcase.pointCalc.query, icon: <Sparkles size={16} /> },
    { title: t.showcase.blueCardSalary.title, query: t.showcase.blueCardSalary.query, icon: <Database size={16} /> },
    { title: t.showcase.studentWork.title, query: t.showcase.studentWork.query, icon: <FileSearch size={16} /> }
  ];

  return (
    <div className="ml-11 mt-2">
      <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
        <SparklesIcon size={14} className="text-accent" />
        {t.showcaseExamples}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {starters.map((s, i) => (
          <button 
            key={i}
            onClick={() => onSelect(s.query)}
            className="text-left p-3 rounded-xl border border-slate-200/80 dark:border-slate-700/50 bg-slate-50/50 dark:bg-slate-800/20 hover:bg-slate-100 dark:hover:bg-slate-800/60 transition-colors group"
          >
            <div className="flex items-center gap-2 text-slate-700 dark:text-slate-300 font-medium mb-1 group-hover:text-accent transition-colors">
              {s.icon} <span className="text-sm">{s.title}</span>
            </div>
            <div className="text-xs text-slate-500 line-clamp-2">"{s.query}"</div>
          </button>
        ))}
      </div>
    </div>
  );
}

function TraceLoader({ currentStatus }: { currentStatus?: string }) {
  const { t } = useTranslation();
  const statusMap: Record<string, number> = {
    'analyzing': 0,
    'retrieving': 1,
    'extracting': 2,
    'synthesizing': 3
  };

  const step = currentStatus ? (statusMap[currentStatus] ?? 0) : 0;

  const steps = [
    { label: t.analyzing || "Analyzing intent & parameters", icon: <SparklesIcon size={14} /> },
    { label: t.retrieving || "Querying official vector database", icon: <Database size={14} /> },
    { label: t.extracting || "Extracting legal requirements", icon: <FileSearch size={14} /> },
    { label: t.synthesizing || "Synthesizing answer", icon: <Bot size={14} /> }
  ];

  return (
    <div className="flex flex-col gap-3 py-2">
      {steps.map((s, i) => (
        <div key={i} className={`flex items-center gap-3 text-sm transition-all duration-500 ${step >= i ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}`}>
          <div className={`p-1.5 rounded-lg flex items-center justify-center ${step === i ? 'bg-accent/10 dark:bg-accent/20 text-accent animate-pulse' : step > i ? 'bg-green-100 dark:bg-green-500/10 text-green-600 dark:text-green-400' : 'bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500'}`}>
            {s.icon}
          </div>
          <span className={`${step === i ? 'text-slate-700 dark:text-slate-300' : step > i ? 'text-slate-500 dark:text-slate-400' : 'text-slate-400 dark:text-slate-600'}`}>
            {s.label}
          </span>
          {step === i && <Loader2 size={12} className="ml-auto text-slate-400 dark:text-slate-500 animate-spin" />}
        </div>
      ))}
    </div>
  );
}
