import { useState, useEffect } from 'react';
import { FileText, ExternalLink, CheckCircle, Shield, Info, AlertCircle, Loader2 } from 'lucide-react';
import { useTranslation } from '../translations';

interface Source {
  title: string;
  url: string;
  authority_level: 'official' | 'semi_official' | 'third_party';
  last_fetched: string;
  visa_types: string[];
}

export function DocumentsPage() {
  const { t } = useTranslation();
  const [sources, setSources] = useState<Source[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSources = async () => {
      try {
        setIsLoading(true);
        // Using the same API key logic as chat (assuming it's in env or handled by proxy)
        const response = await fetch('/query/sources', {
          headers: {
            'X-API-Key': import.meta.env.VITE_API_KEY || 'demo-key'
          }
        });
        
        if (!response.ok) throw new Error('Failed to fetch knowledge base sources');
        
        const data = await response.ok ? await response.json() : [];
        setSources(data);
      } catch (err) {
        console.error('Error fetching sources:', err);
        setError('Could not load knowledge base sources. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchSources();
  }, []);

  const getAuthorityBadge = (level: string) => {
    switch (level) {
      case 'official':
        return (
          <span className="text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1 bg-blue-500/10 text-blue-400 border border-blue-500/20 font-bold uppercase tracking-wider">
            <Shield size={10} /> Official
          </span>
        );
      case 'semi_official':
        return (
          <span className="text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold uppercase tracking-wider">
            <CheckCircle size={10} /> Verified
          </span>
        );
      default:
        return (
          <span className="text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1 bg-slate-500/10 text-slate-400 border border-slate-500/20 font-bold uppercase tracking-wider">
            <Info size={10} /> Third Party
          </span>
        );
    }
  };

  const formatDate = (isoStr: string) => {
    try {
      return new Date(isoStr).toLocaleDateString(undefined, { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
      });
    } catch {
      return 'Recently';
    }
  };

  return (
    <div className="flex-1 glass-panel p-8 bg-slate-900/40 rounded-2xl border border-slate-800/50 shadow-2xl overflow-y-auto w-full">
      <div className="flex justify-between items-center mb-10">
        <div>
          <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
            <div className="p-2 bg-accent/20 rounded-lg text-accent">
              <FileText size={24} />
            </div>
            {t.library}
          </h2>
          <p className="text-slate-400 text-sm mt-2 max-w-2xl">
            Our RAG system is built on real-time indexed legislation and official guidelines. 
            Below are the primary sources currently in our knowledge base.
          </p>
        </div>
      </div>

      {isLoading ? (
        <div className="flex flex-col items-center justify-center py-20 gap-4">
          <Loader2 size={40} className="text-accent animate-spin" />
          <p className="text-slate-400">Loading indexed sources...</p>
        </div>
      ) : error ? (
        <div className="flex flex-col items-center justify-center py-20 gap-4 text-center">
          <div className="p-4 bg-red-500/10 rounded-full text-red-400">
            <AlertCircle size={32} />
          </div>
          <p className="text-slate-300 font-medium">{error}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sources.map((source, i) => (
            <a 
              key={i} 
              href={source.url} 
              target="_blank" 
              rel="noopener noreferrer"
              className="glass-panel p-6 flex flex-col justify-between bg-slate-800/40 hover:bg-slate-800/60 border border-slate-700/50 hover:border-accent/30 transition-all cursor-pointer group hover:-translate-y-1"
            >
              <div>
                <div className="flex items-start justify-between mb-4">
                  {getAuthorityBadge(source.authority_level)}
                  <ExternalLink size={16} className="text-slate-500 group-hover:text-accent transition-colors" />
                </div>
                
                <h3 className="font-semibold text-slate-100 mb-2 line-clamp-2 leading-tight group-hover:text-accent transition-colors">
                  {source.title || 'Untitled Source'}
                </h3>
                
                <p className="text-xs text-slate-500 break-all line-clamp-1 mb-4">
                  {source.url}
                </p>
              </div>

              <div className="pt-4 border-t border-slate-700/50 flex flex-wrap gap-2 items-center justify-between">
                <div className="flex flex-wrap gap-1">
                  {source.visa_types.slice(0, 2).map((v, j) => (
                    <span key={j} className="text-[9px] px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-400 border border-slate-600/30">
                      {v.toUpperCase()}
                    </span>
                  ))}
                  {source.visa_types.length > 2 && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-700/50 text-slate-400 border border-slate-600/30">
                      +{source.visa_types.length - 2}
                    </span>
                  )}
                </div>
                <div className="text-[10px] text-slate-500 italic">
                  Indexed: {formatDate(source.last_fetched)}
                </div>
              </div>
            </a>
          ))}
          
          {sources.length === 0 && (
            <div className="col-span-full py-12 text-center text-slate-500 border-2 border-dashed border-slate-800 rounded-xl">
              No sources found in the knowledge base yet.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
