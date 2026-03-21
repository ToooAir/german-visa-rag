import { CheckCircle2, Circle, AlertCircle, RefreshCw, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useChatStore } from '../stores/chatStore';
import { useTranslation } from '../translations';

export function InsightsPanel({ className = '' }: { className?: string }) {
  const { checklist, requirements, activeVisaCategory, resetProgress } = useChatStore();
  const { t } = useTranslation();

  // Map category to translation
  const getCategoryTitle = (cat: string | null) => {
    if (!cat) return '';
    const key = cat.toLowerCase().replace(/\s+/g, '');
    if (key === 'chancenkarte') return t.chancenkarte;
    if (key === 'skilledworker' || key === 'work_visa') return t.skilledWorker;
    if (key === 'bluecard' || key === 'blue_card') return t.blueCard;
    if (key === 'studyvisa' || key === 'student_visa') return t.studyVisa;
    return cat;
  };

  const titlePrefix = getCategoryTitle(activeVisaCategory);

  // Map checklist titles
  const translateChecklistTitle = (title: string) => {
    const lower = title.toLowerCase();
    if (lower.includes('eligibility')) return t.criteria.eligibility;
    if (lower.includes('point calculation')) return t.pointsReq;
    if (lower.includes('document')) return t.docChecklist;
    if (lower.includes('embassy') || lower.includes('appointment')) return t.embassyAppt;
    if (lower.includes('approval')) return t.approval;
    return title;
  };

  // Map requirement labels
  const translateRequirementLabel = (label: string) => {
    const key = label.toLowerCase().replace(/\s+/g, '');
    if (key === 'language') return t.criteria.language;
    if (key === 'workexperience' || key === 'experience') return t.criteria.experience;
    if (key === 'age') return t.criteria.age;
    if (key === 'qualifications') return t.criteria.qualifications;
    return label;
  };

  return (
    <div className={`flex flex-col gap-4 h-full overflow-y-auto pr-1 custom-scrollbar ${className}`}>
      {/* Progress Checklist */}
      <div className="glass-panel p-5 bg-white/50 dark:bg-panel relative">
        <div className="flex items-center justify-between mb-4 px-1">
          <h3 className="text-[15px] font-semibold text-slate-800 dark:text-slate-200">
            {titlePrefix} {t.progress}
          </h3>
          <button
            onClick={resetProgress}
            className="p-1.5 text-slate-400 hover:text-accent hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-all cursor-pointer group"
            title={t.resetProgress}
          >
            <RefreshCw size={14} className="group-hover:rotate-180 transition-transform duration-500" />
          </button>
        </div>
        <div className="space-y-3">
          <AnimatePresence mode="popLayout">
            {checklist.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center gap-3 group"
              >
                <div className={`shrink-0 ${item.status === 'completed' ? 'text-green-500' : item.status === 'current' ? 'text-accent' : 'text-slate-300 dark:text-slate-600'}`}>
                  {item.status === 'completed' ? <CheckCircle2 size={18} /> : item.status === 'current' ? (
                    <div className="relative">
                      <Circle size={18} className="animate-pulse" />
                      <div className="absolute inset-0 m-auto w-1.5 h-1.5 bg-accent rounded-full" />
                    </div>
                  ) : <Circle size={18} />}
                </div>
                <div className="flex flex-col min-w-0">
                  <span className={`text-sm font-medium truncate ${item.status === 'completed' ? 'text-slate-400 line-through' : item.status === 'current' ? 'text-slate-800 dark:text-slate-100' : 'text-slate-500'}`}>
                    {translateChecklistTitle(item.title)}
                  </span>
                  <div className="flex items-center gap-2">
                    {item.status === 'current' && item.subtitle && (
                      <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-[10px] text-accent font-medium uppercase tracking-wider"
                      >
                        {item.subtitle}
                      </motion.span>
                    )}
                    {item.autoDetected && (
                      <div className="flex items-center gap-1 text-[10px] text-amber-500/80 font-medium">
                        <Sparkles size={10} />
                        <span>{t.autoDetected}</span>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>

      {/* Requirement Summary */}
      <div className="glass-panel p-5 bg-white/50 dark:bg-panel">
        <h3 className="text-[15px] font-semibold text-slate-800 dark:text-slate-200 mb-4 px-1">{t.reqSummary}</h3>

        <div className="space-y-4">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-3 bg-slate-50/80 dark:bg-slate-800/40 rounded-xl border border-slate-200/50 dark:border-slate-700/50"
          >
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">{t.pointsReq}:</div>
            <div className="text-sm font-medium text-slate-800 dark:text-slate-200 flex justify-between items-center">
              <span>6+ Points</span>
              <span className="text-xs text-slate-500 bg-slate-200/50 dark:bg-slate-800 px-2 py-0.5 rounded-full">(0% {t.met})</span>
            </div>
          </motion.div>

          <div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-2 px-1">{t.mainCriteria}:</div>
            <ul className="space-y-2 text-sm text-slate-700 dark:text-slate-300">
              <AnimatePresence mode="popLayout">
                {requirements.map((req, index) => (
                  <motion.li
                    key={req.id || index}
                    initial={{ opacity: 0, x: 5 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex justify-between items-center p-2 rounded-lg hover:bg-slate-50/50 dark:hover:bg-slate-800/30 transition-colors group"
                  >
                    <span className="text-slate-500 dark:text-slate-400">
                      {translateRequirementLabel(req.label)}
                    </span>
                    <div className="flex items-center gap-2">
                      {req.status === 'warning' && <AlertCircle size={14} className="text-amber-500" />}
                      <span className={`px-2 py-0.5 rounded-md text-[11px] font-medium ${req.status === 'required' ? 'bg-green-100 dark:bg-green-500/10 text-green-600 dark:text-green-400' :
                          req.status === 'warning' ? 'bg-amber-100 dark:bg-amber-500/10 text-amber-600 dark:text-amber-400' :
                            'bg-slate-100 dark:bg-slate-800 text-slate-500'
                        }`}>
                        {req.value}
                      </span>
                    </div>
                  </motion.li>
                ))}
              </AnimatePresence>
            </ul>
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-6 p-4 rounded-xl bg-accent/5 border border-accent/10 flex items-start gap-3"
        >
          <div className="p-1.5 bg-accent/10 rounded-lg text-accent mt-0.5">
            <CheckCircle2 size={16} />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-accent mb-1">{t.criteria.eligibility}</h4>
            <p className="text-[12px] text-slate-600 dark:text-slate-400 leading-relaxed">
              {t.eligibilityDesc}
            </p>
          </div>
        </motion.div>
      </div>

    </div>
  );
}
