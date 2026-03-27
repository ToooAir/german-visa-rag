import { useState, useEffect } from 'react';
import { FileText, ExternalLink, CheckCircle, Shield, Info, Loader2 } from 'lucide-react';
import { useTranslation } from '../translations';
import { useChatStore } from '../stores/chatStore';

export function DocumentsPage() {
  const { t, language } = useTranslation();
  const { allSources, isSourcesLoading, fetchSources } = useChatStore();
  const [visibleCount, setVisibleCount] = useState(20);

  useEffect(() => {
    fetchSources();
  }, [fetchSources]);

  // Intersection Observer for Lazy Loading
  useEffect(() => {
    if (isSourcesLoading || allSources.length <= visibleCount) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setTimeout(() => {
            setVisibleCount((prev) => Math.min(prev + 20, allSources.length));
          }, 100);
        }
      },
      { threshold: 0.1 }
    );

    const sentinel = document.getElementById('load-more-sentinel');
    if (sentinel) observer.observe(sentinel);

    return () => observer.disconnect();
  }, [isSourcesLoading, allSources.length, visibleCount]);

  const getAuthorityBadge = (level: string) => {
    switch (level) {
      case 'official':
        return (
          <span className="text-xs px-2.5 py-0.5 rounded-full flex items-center gap-1.5 bg-blue-500/10 text-blue-400 border border-blue-500/20 font-bold uppercase tracking-wider">
            <Shield size={12} /> {t.official || 'Official'}
          </span>
        );
      case 'semi_official':
        return (
          <span className="text-xs px-2.5 py-0.5 rounded-full flex items-center gap-1.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold uppercase tracking-wider">
            <CheckCircle size={12} /> {t.verified || 'Verified'}
          </span>
        );
      default:
        return (
          <span className="text-xs px-2.5 py-0.5 rounded-full flex items-center gap-1.5 bg-slate-500/10 text-slate-400 border border-slate-500/20 font-bold uppercase tracking-wider">
            <Info size={12} /> {t.thirdParty || 'Third Party'}
          </span>
        );
    }
  };

  const formatDate = (isoStr: string) => {
    try {
      const date = new Date(isoStr);
      const locale = language === 'zh-TW' ? 'zh-TW' : language === 'de' ? 'de-DE' : 'en-GB';
      return date.toLocaleDateString(locale, {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    } catch {
      return t.recently || 'Recently';
    }
  };

  return (
    <div className="flex-1 glass-panel px-7 pt-[calc(6rem+env(safe-area-inset-top))] pb-[calc(2rem+env(safe-area-inset-bottom))] sm:p-8 bg-white/70 dark:bg-slate-900/40 lg:rounded-2xl rounded-none border-0 lg:border border-slate-200 dark:border-slate-800/50 lg:shadow-2xl overflow-y-auto w-full">
      <div className="flex justify-between items-center mb-10">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-4">
            <div className="p-2 bg-accent/20 rounded-lg text-accent">
              <FileText size={24} />
            </div>
            {t.library}
          </h2>
          <p className="text-slate-600 dark:text-slate-400 text-base sm:text-sm mt-4 max-w-2xl leading-relaxed">
            {t.libraryDesc}
          </p>
        </div>
      </div>

      {isSourcesLoading && allSources.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 gap-4">
          <Loader2 size={40} className="text-accent animate-spin" />
          <p className="text-slate-400">Loading indexed sources...</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {allSources.slice(0, visibleCount).map((source, i) => {
              const hostname = new URL(source.url).hostname.replace('www.', '');
              return (
                <a
                  key={i}
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="glass-panel p-5 sm:p-5 flex flex-col justify-between bg-white/5 dark:bg-slate-800/15 hover:bg-white/10 dark:hover:bg-slate-800/25 border border-slate-200/40 dark:border-slate-800/50 hover:border-accent/30 transition-all cursor-pointer group hover:-translate-y-1 relative overflow-hidden min-h-[190px] h-auto min-w-0"
                >
                  {/* Background Watermark Icon */}
                  <div className="absolute -right-2 -bottom-2 opacity-[0.02] dark:opacity-[0.04] group-hover:opacity-[0.06] transition-opacity pointer-events-none transform rotate-12">
                    <FileText size={80} />
                  </div>

                  <div className="relative z-10">
                    <div className="flex items-start justify-between mb-3.5">
                      {getAuthorityBadge(source.authority_level || 'third_party')}
                      <ExternalLink size={14} className="text-slate-500 group-hover:text-accent transition-colors" />
                    </div>

                    <h3 className="text-base sm:text-sm font-bold text-slate-800 dark:text-slate-100 mb-2.5 line-clamp-2 leading-snug group-hover:text-accent transition-colors">
                      {source.title || 'Untitled Source'}
                    </h3>

                    <div className="flex items-center gap-2 text-xs sm:text-[10px] text-slate-500 font-medium opacity-80 group-hover:opacity-100 transition-opacity">
                      <div className="w-4 h-4 rounded-sm overflow-hidden bg-slate-100/50 dark:bg-slate-800/50 flex items-center justify-center">
                        <img
                          src={`https://www.google.com/s2/favicons?domain=${hostname}&sz=32`}
                          alt=""
                          className="w-3 h-3 grayscale group-hover:grayscale-0 transition-all"
                        />
                      </div>
                      <span className="truncate">{hostname}</span>
                    </div>
                  </div>

                  <div className="mt-5 pt-4 border-t border-slate-100/50 dark:border-slate-800/50 flex flex-wrap gap-1.5 items-center justify-between relative z-10">
                    <div className="flex flex-wrap gap-1">
                      {source.visa_types?.slice(0, 1).map((v, j) => (
                        <span key={j} className="text-[10px] sm:text-[8px] px-2 py-0.5 rounded bg-accent/5 dark:bg-accent/10 text-accent font-bold border border-accent/10 uppercase tracking-tighter">
                          {v.replace('_', ' ')}
                        </span>
                      ))}
                      {source.visa_types && source.visa_types.length > 1 && (
                        <span className="text-[10px] sm:text-[8px] px-2 py-0.5 rounded bg-slate-100/50 dark:bg-slate-800/30 text-slate-500 font-bold border border-slate-200/50 dark:border-slate-700/50">
                          +{source.visa_types.length - 1}
                        </span>
                      )}
                    </div>
                    <div className="text-[10px] sm:text-[9px] text-slate-400 dark:text-slate-500 font-mono tracking-tighter">
                      {formatDate(source.last_fetched || '')}
                    </div>
                  </div>
                </a>
              );
            })}

            {allSources.length === 0 && !isSourcesLoading && (
              <div className="col-span-full py-12 text-center text-slate-500 border-2 border-dashed border-slate-800 rounded-xl">
                No sources found in the knowledge base yet.
              </div>
            )}
          </div>

          {/* Sentinel for Lazy Loading */}
          {visibleCount < allSources.length && (
            <div id="load-more-sentinel" className="h-20 flex items-center justify-center mt-8">
              <Loader2 size={24} className="text-accent animate-spin opacity-50" />
            </div>
          )}
        </>
      )}
    </div>
  );
}
