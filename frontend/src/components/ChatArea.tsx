import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MoreVertical, Send, ShieldCheck, ExternalLink, Loader2, Sparkles,
  Database, FileSearch, Trash2, ChevronDown, Briefcase,
  Award, BadgeCheck, GraduationCap
} from 'lucide-react';
import { useChatStore, Message, Source } from '../stores/chatStore';
import { useTranslation, Translation } from '../translations';

// Mapping for visa categories: Store ID -> Display Metadata
const VISA_METADATA: Record<string, { labelKey: string, icon: React.ElementType }> = {
  chancenkarte: { labelKey: 'chancenkarte', icon: Briefcase },
  skilledWorker: { labelKey: 'skilledWorker', icon: Award },
  blueCard: { labelKey: 'blueCard', icon: BadgeCheck },
  studyVisa: { labelKey: 'studyVisa', icon: GraduationCap }
};

export function ChatArea({ className = '' }: { className?: string }) {
  const { messages, sendMessage, isLoading, activeVisaCategory, setActiveVisaCategory } = useChatStore();
  const { t } = useTranslation();
  const [input, setInput] = useState('');
  const [isSelectorOpen, setIsSelectorOpen] = useState(false);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const handleNewSession = () => {
    useChatStore.getState().newSession();
    setShowConfirmModal(false);
  };

  // Auto-scroll to bottom
  useEffect(() => {
    // Only auto-scroll if there are actual user/assistant messages (length > 1)
    // This allows the welcome message to be shown at the top initially.
    if (scrollContainerRef.current && messages.length > 1) {
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

  // Get current metadata
  const currentMeta = VISA_METADATA[activeVisaCategory || 'chancenkarte'] || VISA_METADATA.chancenkarte;
  const ActiveIcon = currentMeta.icon;

  return (
    <div className={`flex flex-col h-full bg-white/40 dark:bg-slate-900/40 lg:rounded-2xl rounded-none relative lg:border border-0 border-slate-200/50 dark:border-slate-800/50 lg:shadow-2xl overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 lg:pt-3 pt-[calc(4.5rem+env(safe-area-inset-top))] border-b border-slate-200/50 dark:border-slate-800/50 bg-white/50 dark:bg-slate-900/50 backdrop-blur-md z-20">
        <div className="flex items-center gap-3 relative">
          <div className="flex flex-col">
            <button
              onClick={() => setIsSelectorOpen(!isSelectorOpen)}
              className="flex items-center gap-2 px-1.5 py-1 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors group lg:px-2 lg:py-1.5"
            >
              <div className="p-1 bg-accent/10 rounded-lg text-accent shrink-0">
                <ActiveIcon size={16} className="lg:w-[18px] lg:h-[18px]" />
              </div>
              <span className="text-base lg:text-lg font-bold text-slate-800 dark:text-slate-100 truncate max-w-[120px] sm:max-w-none">
                {activeVisaCategory ? (t[currentMeta.labelKey as keyof Translation] as string) : t.chatHeader}
              </span>
              <ChevronDown size={14} className={`text-slate-400 transition-transform duration-300 lg:w-[16px] lg:h-[16px] ${isSelectorOpen ? 'rotate-180' : ''}`} />
              {isLoading && <Loader2 className="w-4 h-4 animate-spin text-accent ml-1" />}
            </button>
          </div>

          {/* Visa Selector Dropdown */}
          <AnimatePresence>
            {isSelectorOpen && (
              <>
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="fixed inset-0 z-30"
                  onClick={() => setIsSelectorOpen(false)}
                />
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  className="absolute top-[110%] left-0 w-64 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-2xl p-2 z-40 backdrop-blur-xl"
                >
                  <div className="flex flex-col gap-1">
                    {Object.entries(VISA_METADATA).map(([id, meta]) => {
                      const Icon = meta.icon;
                      const isActive = activeVisaCategory === id;
                      return (
                        <button
                          key={id}
                          onClick={() => {
                            setActiveVisaCategory(id);
                            setIsSelectorOpen(false);
                          }}
                          className={`flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 ${isActive
                            ? 'bg-accent/10 text-accent font-bold'
                            : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200'
                            }`}
                        >
                          <div className={`p-1.5 rounded-lg ${isActive ? 'bg-accent/20' : 'bg-slate-100 dark:bg-slate-800 text-slate-400'}`}>
                            <Icon size={16} />
                          </div>
                          <span className="flex-1 text-left text-sm">{(t[meta.labelKey as keyof Translation] as string)}</span>
                          {isActive && <div className="w-1.5 h-1.5 rounded-full bg-accent shadow-[0_0_8px_rgba(var(--accent-rgb),0.5)]" />}
                        </button>
                      );
                    })}
                  </div>
                </motion.div>
              </>
            )}
          </AnimatePresence>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowConfirmModal(true)}
            className="flex items-center gap-2 px-2 py-1.5 lg:px-3 rounded-xl text-slate-500 dark:text-slate-400 hover:text-rose-500 dark:hover:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-500/10 transition-all duration-200 group border border-transparent hover:border-rose-200 dark:hover:border-rose-500/20"
            title={t.newSession}
          >
            <Trash2 size={16} className="group-hover:scale-110 transition-transform" />
            <span className="hidden sm:inline text-xs font-semibold">{t.newSession}</span>
          </button>
        </div>
      </div>

      <AnimatePresence>
        {showConfirmModal && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowConfirmModal(false)}
              className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-sm glass-panel bg-white/95 dark:bg-slate-900/95 border border-slate-200 dark:border-slate-800 p-6 rounded-3xl shadow-2xl overflow-hidden"
            >
              <div className="absolute -top-12 -right-12 w-32 h-32 bg-rose-500/10 dark:bg-rose-500/5 rounded-full blur-3xl pointer-events-none" />

              <div className="flex items-center gap-4 mb-5 relative z-10">
                <div className="w-12 h-12 rounded-2xl bg-rose-500/10 flex items-center justify-center text-rose-500 shadow-inner">
                  <Trash2 size={24} />
                </div>
                <div>
                  <h3 className="text-lg font-extrabold text-slate-800 dark:text-slate-100">
                    {t.newSessionConfirm || t.newSession + '?'}
                  </h3>
                  <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mt-0.5 opacity-70">
                    {t.irreversible || 'Irreversible Action'}
                  </p>
                </div>
              </div>

              <p className="text-slate-600 dark:text-slate-400 text-sm leading-relaxed mb-8 relative z-10">
                {t.newSessionDesc || '這將清除當前的對話紀錄，並重置所有簽證進度評等指標。'}
              </p>

              <div className="flex gap-3 relative z-10">
                <button
                  onClick={() => setShowConfirmModal(false)}
                  className="flex-1 px-4 py-3 rounded-2xl bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 font-bold text-xs hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                >
                  {t.cancel || '取消'}
                </button>
                <button
                  onClick={handleNewSession}
                  className="flex-1 px-4 py-3 rounded-2xl bg-rose-500 text-white font-bold text-xs hover:bg-rose-600 transition-colors shadow-lg shadow-rose-500/20 active:scale-95"
                >
                  {t.confirmReset || '確認重置'}
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Message List */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto p-3 sm:p-4 pb-[calc(11.5rem+env(safe-area-inset-bottom))] pt-4 lg:pt-4 flex flex-col gap-5 lg:gap-6 relative"
      >
        <AnimatePresence initial={false}>
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
      <div className="absolute bottom-0 w-full bg-gradient-to-t from-background via-background to-transparent pb-[calc(1rem+env(safe-area-inset-bottom))] lg:pb-6 pt-6 lg:pt-8 z-20">
        <div className="relative glass-panel bg-white/60 dark:bg-slate-800/60 backdrop-blur-xl border border-slate-200 dark:border-slate-700 mx-3 sm:mx-4 flex items-center shadow-2xl shadow-black/5 dark:shadow-black/50 overflow-hidden">
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
            className="w-full bg-transparent text-slate-800 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 py-5 pl-5 pr-14 outline-none text-base lg:text-base resize-none overflow-hidden"
            rows={1}
            style={{ minHeight: '64px' }}
          />
          <div className="absolute right-3 flex items-center gap-1">
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
            <span className="font-medium text-slate-800 dark:text-slate-200 text-sm">{(t.you as string)}</span>
          </div>
          <div className="flex items-center gap-2 text-slate-400 dark:text-slate-500 text-xs">
            <MoreVertical size={14} className="cursor-pointer hover:text-slate-600 dark:hover:text-slate-300" />
          </div>
        </div>
        <div className="text-slate-700 dark:text-slate-300 text-base lg:text-base leading-relaxed whitespace-pre-wrap">
          {message.content}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className="glass-panel p-6 flex flex-col gap-4 w-full bg-white dark:bg-panel border-l-4 border-l-accent shadow-lg relative"
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-accent/20 flex items-center justify-center text-accent ring-1 ring-accent/30 shadow-inner">
            <Sparkles size={20} />
          </div>
          <span className="font-bold text-slate-900 dark:text-white text-base tracking-wide">{(t.personaName as string) || 'Visa Assistant'}</span>
        </div>
      </div>

      <div className="text-slate-700 dark:text-slate-300 text-base lg:text-base leading-relaxed ml-0 lg:ml-14 flex flex-col gap-5">
        {/* RAG Traceability Loader */}
        {(isLoading || message.searchQueries) && !message.content && (
          <TraceLoader currentStatus={message.status} searchQueries={message.searchQueries} />
        )}

        {/* Source Grid (Early Visibility) */}
        {message.sources && message.sources.length > 0 && !message.content && (
          <SourceGrid sources={message.sources} />
        )}

        {message.content && (
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

        {/* Source Chips (End of message) */}
        {message.sources && message.sources.length > 0 && message.content && (
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

function SourceGrid({ sources }: { sources: Source[] }) {
  const uniqueSources = Array.from(new Map(sources.map(src => [src.url, src])).values());

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 mt-2 mb-4"
    >
      {uniqueSources.slice(0, 3).map((src, i) => (
        <SourceCard key={i} source={src} index={i} />
      ))}
      {uniqueSources.length > 3 && (
        <div className="flex items-center justify-center p-3 rounded-xl border border-dashed border-slate-200 dark:border-slate-700 text-xs text-slate-500 bg-slate-50/30 dark:bg-slate-800/20">
          +{uniqueSources.length - 3} more sources
        </div>
      )}
    </motion.div>
  );
}

function SourceCard({ source, index }: { source: Source, index: number }) {
  const hostname = new URL(source.url).hostname.replace('www.', '');

  return (
    <motion.a
      whileHover={{ y: -2, boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.1 }}
      className="flex flex-col gap-2 p-3 rounded-xl border border-slate-200/80 dark:border-slate-700/80 bg-white dark:bg-slate-800/40 hover:bg-slate-50 dark:hover:bg-slate-700/60 transition-all group"
    >
      <div className="flex items-center gap-2">
        <div className="w-5 h-5 rounded overflow-hidden bg-slate-100 dark:bg-slate-700 flex items-center justify-center">
          <img src={`https://www.google.com/s2/favicons?domain=${hostname}&sz=32`} alt="" className="w-3 h-3" />
        </div>
        <span className="text-[10px] font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider truncate">{hostname}</span>
      </div>
      <h4 className="text-xs font-semibold text-slate-800 dark:text-slate-200 line-clamp-2 leading-tight group-hover:text-accent transition-colors">
        {source.title}
      </h4>
    </motion.a>
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
    { title: t.showcase.studentWork.title, query: t.showcase.studentWork.query, icon: <FileSearch size={16} /> },
    { title: t.showcase.skilledWorker.title, query: t.showcase.skilledWorker.query, icon: <Award size={16} /> }
  ];

  return (
    <div className="ml-0 mt-4">
      <div className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-5 flex items-center gap-2.5 lg:ml-0">
        <Sparkles size={16} className="text-accent" />
        {(t.showcaseExamples as string)}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {starters.map((s, i) => (
          <button
            key={i}
            onClick={() => onSelect(s.query)}
            className="text-left p-4 rounded-2xl border border-slate-200/80 dark:border-slate-700/50 bg-slate-50/50 dark:bg-slate-800/20 hover:bg-slate-100 dark:hover:bg-slate-800/60 transition-all hover:border-accent/30 group shadow-sm hover:shadow-md"
          >
            <div className="flex items-center gap-2.5 text-slate-800 dark:text-slate-200 font-bold mb-1.5 group-hover:text-accent transition-colors">
              {s.icon} <span className="text-base">{s.title}</span>
            </div>
            <div className="text-sm text-slate-500 line-clamp-2 leading-snug">"{s.query}"</div>
          </button>
        ))}
      </div>
    </div>
  );
}

function TraceLoader({ currentStatus, searchQueries }: { currentStatus?: string, searchQueries?: string[] }) {
  const { t } = useTranslation();
  const statusMap: Record<string, number> = {
    'analyzing': 0,
    'retrieving': 1,
    'extracting': 2,
    'synthesizing': 3
  };

  const step = currentStatus ? (statusMap[currentStatus] ?? 0) : 0;

  const steps = [
    { label: t.analyzing || "Analyzing intent & parameters", icon: <Sparkles size={14} />, key: 'analyzing' },
    { label: t.retrieving || "Querying official vector database", icon: <Database size={14} />, key: 'retrieving' },
    { label: t.extracting || "Extracting legal requirements", icon: <FileSearch size={14} />, key: 'extracting' },
    { label: t.synthesizing || "Synthesizing answer", icon: <Sparkles size={14} />, key: 'synthesizing' }
  ];

  return (
    <div className="flex flex-col gap-4 py-2">
      {steps.map((s, i) => (
        <div key={i} className={`flex flex-col gap-2 transition-all duration-500 ${step >= i ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2'}`}>
          <div className="flex items-center gap-3 text-sm">
            <div className={`p-1.5 rounded-lg flex items-center justify-center ${step === i ? 'bg-accent/10 dark:bg-accent/20 text-accent animate-pulse' : step > i ? 'bg-green-100 dark:bg-green-500/10 text-green-600 dark:text-green-400' : 'bg-slate-100 dark:bg-slate-800 text-slate-400 dark:text-slate-500'}`}>
              {s.icon}
            </div>
            <span className={`${step === i ? 'text-slate-700 dark:text-slate-300 font-medium' : step > i ? 'text-slate-500 dark:text-slate-400' : 'text-slate-400 dark:text-slate-600'}`}>
              {s.label}
            </span>
            {step === i && <Loader2 size={12} className="ml-auto text-slate-400 dark:text-slate-500 animate-spin" />}
          </div>

          {/* Show search queries if in retrieving step OR if they exist and we are further along */}
          {s.key === 'retrieving' && searchQueries && searchQueries.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="ml-10 flex flex-col gap-1.5 overflow-hidden"
            >
              {searchQueries.map((q, idx) => (
                <div key={idx} className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-2">
                  <div className="w-1 h-1 rounded-full bg-slate-300 dark:bg-slate-600" />
                  <span className="italic">"{q}"</span>
                </div>
              ))}
            </motion.div>
          )}
        </div>
      ))}
    </div>
  );
}
