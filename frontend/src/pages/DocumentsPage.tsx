import { FileText, Download, CheckCircle } from 'lucide-react';

export function DocumentsPage() {
  const docs = [
    { name: 'Chancenkarte_Federal_Law_2024.pdf', size: '1.2 MB', date: 'Jan 15, 2024', status: 'official' },
    { name: 'Skilled_Worker_Immigration_Act.pdf', size: '3.5 MB', date: 'Mar 01, 2024', status: 'official' },
    { name: 'Blue_Card_Requirements_EU.pdf', size: '840 KB', date: 'Feb 10, 2024', status: 'official' },
    { name: 'Student_Visa_Guidelines.pdf', size: '2.1 MB', date: 'Jan 05, 2024', status: 'official' },
  ];

  return (
    <div className="flex-1 glass-panel p-8 bg-slate-900/40 rounded-2xl border border-slate-800/50 shadow-2xl overflow-y-auto w-full">
      <div className="flex justify-between items-center mb-8">
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-2xl font-bold text-slate-100">Knowledge Library</h2>
      </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {docs.map((doc, i) => (
          <div key={i} className="glass-panel p-5 flex flex-col gap-4 bg-slate-800/40 hover:bg-slate-800/60 transition-colors cursor-pointer group">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-slate-700/50 rounded-xl text-accent group-hover:scale-110 transition-transform">
                  <FileText size={24} />
                </div>
                <div>
                  <h3 className="font-medium text-slate-200 truncate max-w-[200px]">{doc.name}</h3>
                  <div className="text-xs text-slate-400 mt-1">{doc.size} • Uploaded {doc.date}</div>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-between pt-4 border-t border-slate-700/50">
              <span className={`text-xs px-2 py-1 rounded flex items-center gap-1.5 bg-blue-500/10 text-blue-400 border border-blue-500/20`}>
                <CheckCircle size={12} />
                OFFICIAL SOURCE
              </span>
              <button className="text-slate-400 hover:text-accent p-1 transition-colors">
                <Download size={16} />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
