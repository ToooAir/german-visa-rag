import { CheckCircle2, Circle, AlertCircle, RefreshCw, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useChatStore } from '../stores/chatStore';
import { useTranslation } from '../translations';

export function InsightsPanel({ className = '' }: { className?: string }) {
  const { checklist, requirements, activeVisaCategory, resetProgress } = useChatStore();
  const { t, language } = useTranslation();

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
    if (lower.includes('eligibility path')) return t.criteria.eligibilityPath;
    if (lower.includes('eligibility check')) return t.criteria.eligibility;
    if (lower.includes('basic thresholds')) return t.criteria.basicThresholds;
    if (lower.includes('salary threshold')) return t.criteria.salaryThreshold;
    if (lower.includes('points calculation') || lower.includes('points requirements')) return t.pointsReq;
    if (lower.includes('university admission') || lower.includes('zulassung')) return t.criteria.uniAdmission;
    if (lower.includes('job offer') || lower.includes('full-time contract')) return t.criteria.fullTimeContract;
    if (lower.includes('salary check')) return t.salaryCheck;
    if (lower.includes('document')) return t.docChecklist;
    if (lower.includes('embassy') || lower.includes('appointment')) return t.embassyAppt;
    if (lower.includes('approval') || lower.includes('final step')) return t.approval;
    return title;
  };

  const translateRequirementLabel = (label: string, id?: string) => {
    const key = label.toLowerCase().replace(/\s+/g, '').replace(/[^a-z0-9]/g, '');
    let result = label;
    if (key === 'salarythreshold') result = t.criteria.salaryThreshold;
    else if (key === 'age') result = t.criteria.age;
    else if (key === 'language' || key === 'languageproficiency') result = t.criteria.language;
    else if (key === 'experience' || key === 'workexperience') result = t.criteria.experience;
    else if (key === 'qualifications' || key === 'academicqualifications' || key === 'professionalqualifications') result = t.criteria.qualifications;
    else if (key === 'blockedaccount' || key === 'financialproof' || key === 'proofoffinance') result = t.criteria.financialProof;
    else if (key === 'joboffer' || key === 'joboffercontract' || key === 'fulltimecontract' || key === 'germanfulltimecontract') result = t.criteria.fullTimeContract;
    else if (key === 'laborconditions' || key === 'workingconditions') result = t.criteria.laborConditions;
    else if (key === 'age45clause' || key === '45ageclause') result = t.criteria.age45Clause;
    else if (key === 'flexiblelanguage' || key === 'languageflexible' || key === 'languageability') result = t.criteria.flexibleLanguage;
    else if (key === 'academicprofessionalqualifications') result = t.criteria.qualifications;
    else if (key === 'prbonuslanguage' || key === 'prlanguagebonus') result = t.criteria.prLanguageBonus;
    else if (key === 'healthinsurance' || key === 'germanhealthinsurance') result = t.criteria.healthInsurance;
    else if (key === 'prestudyqualifications') result = t.criteria.preStudyQuals;
    else if (key === 'itexemption' || key === 'itexperienceexemption' || key === 'itexperienceexemption3yr') result = t.criteria.itExemption;
    else if (key === 'recognitionpartnership' || key === 'recognitionpartnershipfeg20') result = t.criteria.recognitionPartnership;
    else if (key === 'contractduration' || key === 'contractdurationmin6mo') result = t.criteria.contractDuration;
    else if (key === 'germanresidency6mo' || key === 'germanresidency') result = t.criteria.residencyBonus;
    else if (key === 'partnerbonus') result = t.criteria.partnerBonus;
    else if (key === 'mandatorythresholds' || key === 'header1') {
      if (activeVisaCategory === 'chancenkarte') {
        return language === 'zh-TW' ? '前置要求 (依申請路徑)' : language === 'de' ? 'Voraussetzungen (je nach Pfad)' : 'Prerequisites (Path-dependent)';
      }
      result = t.criteria.basicThresholds;
    }
    else if (key === 'pointsitems' || key === 'pointsitemstarget6' || key === 'header2') result = t.criteria.pointsItems;
    else if (key === 'corerequirement') result = t.criteria.coreRequirement;

    if (id === '1-2' && activeVisaCategory === 'chancenkarte') {
      const suffix = language === 'zh-TW' ? '(僅積分制)' : language === 'de' ? '(nur Punkte-Pfad)' : '(Points Path only)';
      return `${result} ${suffix}`;
    }

    return result;
  };

  const calculatePoints = () => {
    let total = 0;
    requirements.forEach(req => {
      if (req.id?.startsWith('2-') && req.status === 'required') {
        const match = req.value.match(/\+(\d+)/);
        if (match) total += parseInt(match[1], 10);
        else {
           // fallback: try to find just a number before '分' or 'pts'
           const fallbackMatch = req.value.match(/(\d+)\s*(?:分|pts)/i);
           if (fallbackMatch) total += parseInt(fallbackMatch[1], 10);
        }
      }
    });
    return total;
  };

  const checkStatus = (idStart: string) => {
    const list = requirements.filter(r => r.id?.startsWith(idStart));
    if (!list.length) return 0;
    const met = list.filter(r => r.status === 'required').length;
    return Math.round((met / list.length) * 100);
  };

  const getPrimaryThresholds = (cat: string | null) => {
    const cards: Array<{ label: string, value: string, sub: string }> = [];
    
    if (cat === 'work_visa' || cat === 'skilledworker') {
      const prog = checkStatus('1') || checkStatus('2') ? Math.round(((requirements.filter(r => r.id === '1' || r.id === '2').filter(r => r.status === 'required').length) / 2) * 100) : 0;
      cards.push({ 
        label: t.criteria.coreRequirement || 'Core Requirement', 
        value: `${t.criteria.fullTimeContract || 'Full-time Contract'}\n${t.criteria.salaryStandards}`, 
        sub: `(${prog}% ${t.met})` 
      });
    } else if (cat === 'blue_card') {
      const prog = checkStatus('2');
      cards.push({ 
        label: t.criteria.salaryThreshold || 'Salary Threshold', 
        value: `€50,700 (${t.criteria.blueCardGen})\n€45,934.20 (${t.criteria.blueCardIT})`, 
        sub: `(${prog}% ${t.met})` 
      });
    } else if (cat === 'student_visa') {
      cards.push({ label: t.criteria.coreRequirement || 'Core Requirement', value: t.criteria.uniAdmission, sub: `(${checkStatus('4')}% ${t.met})` });
      cards.push({ label: t.criteria.financialGoal || 'Financial Requirement', value: `${t.criteria.yearlyAmount}: €11,904`, sub: `(${checkStatus('1')}% ${t.met})` });
    } else if (cat === 'chancenkarte') {
      // Base progress: considers 1-2 (Language) and 1-3 (Qualifications)
      const baseProgList = requirements.filter(r => r.id === '1-2' || r.id === '1-3');
      const baseProg = baseProgList.length ? Math.round((baseProgList.filter(r => r.status === 'required').length / baseProgList.length) * 100) : 0;
      const pts = calculatePoints();
      cards.push({ label: t.criteria.eligibilityPath || 'Eligibility Path', value: `${t.criteria.pathDirect}\n${t.criteria.pathPoints}`, sub: `(${baseProg}% ${t.met} / ${pts} ${t.criteria.pts})` });
      cards.push({ label: t.criteria.financialGoal || 'Financial Requirement', value: `${t.criteria.yearlyAmount}: €13,092`, sub: `(${requirements.find(r => r.id === '1-1')?.status === 'required' ? 100 : 0}% ${t.met})` });
    } else {
      cards.push({ 
        label: t.criteria.eligibilityPath || 'Eligibility Path', 
        value: t.criteria.pathDirect + ` / 6+ ${t.criteria.pts}`, 
        sub: `(0% ${t.met})` 
      });
    }
    return cards;
  };

  const primaryThresholds = getPrimaryThresholds(activeVisaCategory);

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
            className="flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-bold text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/30 hover:bg-amber-100 dark:hover:bg-amber-900/40 rounded-full border border-amber-200/50 dark:border-amber-800/50 transition-all cursor-pointer group shadow-sm active:scale-95"
            title={t.resetProgress}
          >
            <RefreshCw size={12} className="group-hover:rotate-180 transition-transform duration-500" />
            <span>{t.reset}</span>
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
          {primaryThresholds.map((threshold, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="p-3 bg-slate-50/80 dark:bg-slate-800/40 rounded-xl border border-slate-200/50 dark:border-slate-700/50"
            >
              <div className="flex flex-col gap-1">
                <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">{threshold.label}:</div>
                <div className="space-y-1.5">
                  {threshold.value.split('\n').filter(Boolean).map((line: string, i: number) => (
                    <div key={i} className="flex items-start gap-2">
                      <div className="w-1 h-1 rounded-full bg-accent/40 mt-2 shrink-0" />
                      <span className="text-[13px] font-bold text-slate-800 dark:text-slate-100 leading-tight">
                        {line}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="flex items-center gap-2 mt-2">
                  <div className="text-[10px] font-semibold text-accent bg-accent/10 px-2 py-0.5 rounded-md uppercase tracking-wide">
                    {threshold.sub}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}

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
                    className={`flex justify-between items-start gap-4 p-2 rounded-lg transition-colors group ${req.id?.startsWith('header') ? 'mt-4 mb-1 bg-slate-100/50 dark:bg-slate-800/50 border-l-2 border-indigo-500' : 'hover:bg-slate-50/50 dark:hover:bg-slate-800/30'
                      }`}
                  >
                    <span className={`flex-1 leading-normal ${req.id?.startsWith('header') ? 'text-[11px] font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400' : 'text-slate-500 dark:text-slate-400'
                      }`}>
                      {translateRequirementLabel(req.label, req.id)}
                    </span>
                    {!req.id?.startsWith('header') && (
                      <div className="flex items-center gap-2 pt-[2px] shrink-0">
                        {req.status === 'warning' && <AlertCircle size={14} className="text-amber-500" />}
                        <span className={`min-w-[1.25rem] text-center px-2 py-0.5 rounded-md text-[11px] font-medium ${req.status === 'required' ? 'bg-green-100 dark:bg-green-500/10 text-green-600 dark:text-green-400' :
                          req.status === 'warning' ? 'bg-amber-100 dark:bg-amber-500/10 text-amber-600 dark:text-amber-400' :
                            'bg-slate-100 dark:bg-slate-800 text-slate-500'
                          }`}>
                          {req.value}
                        </span>
                      </div>
                    )}
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
