import { CheckCircle2, Circle, AlertCircle, Sparkles, FileText, CheckSquare } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useChatStore } from '../stores/chatStore';
import { useTranslation } from '../translations';

export function InsightsPanel({ className = '' }: { className?: string }) {
  const { checklist, requirements, activeVisaCategory } = useChatStore();
  const { t, language } = useTranslation();

  // Map category to translation
  const getCategoryTitle = (cat: string | null) => {
    if (!cat) return '';
    const key = cat.toLowerCase().replace(/[\s_-]+/g, '');
    if (key === 'chancenkarte') return t.chancenkarte;
    if (key === 'skilledworker' || key === 'skilled_worker') return t.skilledWorker;
    if (key === 'bluecard' || key === 'blue_card') return t.blueCard;
    if (key === 'studentvisa' || key === 'student_visa' || key === 'studyvisa') return t.studyVisa;
    return cat;
  };

  const titlePrefix = getCategoryTitle(activeVisaCategory);

  // Map checklist titles
  const translateChecklistTitle = (title: string) => {
    switch(title) {
      case 'Basic Thresholds': return t.criteria.basicThresholds || title;
      case 'Points Calculation': return t.criteria.pointsItems || title;
      case 'Document Checklist': return t.docChecklist || title;
      case 'Degree Recognition': return t.criteria.degreeQuals || title;
      case 'High Salary Threshold': return t.criteria.salaryThresholdCheck || title;
      case 'Professional Qualifications': return t.criteria.profQualifications || title;
      case 'Full-time Contract': return t.criteria.employmentContract || title;
      case 'Finance & Language': return t.criteria.finLanguage || title;
      case 'University Admission': return t.criteria.admissionQuals || title;
      default: return title;
    }
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

  const translateRequirementValue = (value: string): string => {
    if (!value) return value;

    // Case: Points with key (e.g. "5_YEARS_EXP|3")
    if (value.includes('|')) {
      const [key, pts] = value.split('|');
      const translatedKey = translateRequirementValue(key);
      const ptsLabel = t.criteria.pts || (language === 'zh-TW' ? '分' : 'pts');
      return `${translatedKey} (+${pts}${ptsLabel})`;
    }

    const key = value.toUpperCase().trim();
    const rv = t.criteria.reqValues;
    const ll = t.criteria.langLevels;

    // Core Status Keys
    if (key === 'TBC') return rv?.tbc || value;
    if (key === 'MET') return rv?.met || value;
    if (key === 'REQUIRED') return rv?.required || rv?.met || value;

    // Language Levels (A1, B1, etc.)
    const langLevelKey = value.toLowerCase();
    if (ll && ll[langLevelKey as keyof typeof ll]) {
      return ll[langLevelKey as keyof typeof ll];
    }

    // Work & Experience
    if (key === '5_YEARS_EXP' || key === '5') return rv?.yearsExp5 || value;
    if (key === '2_YEARS_EXP' || key === '2') return rv?.yearsExp2 || value;
    if (key === 'IT_3Y_EXP' || key === '3_YEARS_IT_EXP') return rv?.itExp3y || value;
    if (key === 'IT_EXP' || key === 'IT_EXP_GENERAL' || key === 'IT') return rv?.itExpGeneral || value;

    // Personal Attributes
    if (key === 'UNDER_35') return rv?.under35 || value;
    if (key === 'UNDER_40') return rv?.under40 || value;
    if (key === 'OVER_45') return rv?.over45 || value;

    // Qualifications
    if (key === 'DEGREE') return rv?.degree || value;
    if (key === 'VOCATIONAL') return rv?.vocational || value;
    if (key === 'PARTIAL_RECOGNITION') return rv?.partialRecognition || value;

    // Legal & Process
    if (key === 'SALARY_MET') return rv?.salaryMet || value;
    if (key === 'SALARY_BELOW') return rv?.salaryBelow || value;
    if (key === 'BA_PASSED') return rv?.baPassed || value;
    if (key === 'BA_PENDING') return rv?.baPending || value;
    if (key === 'CONTRACT_SIGNED') return rv?.contractSigned || value;
    if (key === 'ADMITTED' || key === 'ADMISSION_LETTER') return rv?.admitted || value;
    if (key === 'INSURED' || key === 'INSURANCE_READY') return rv?.insured || value;
    if (key === 'PRE_STUDY_MET') return rv?.preStudyMet || value;

    // Case: LACK_OF_FUNDS (without amount)
    if (key === 'LACK_OF_FUNDS') return rv?.tbc || '待確認';

    // Case: LACK_OF_FUNDS:13092
    if (key.startsWith('LACK_OF_FUNDS:')) {
      const amount = key.split(':')[1]?.trim();
      if (!amount) return rv?.tbc || '待確認';
      const formatted = amount.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
      return (rv?.lackOfFunds || '缺 €{amount}').replace('{amount}', formatted);
    }

    return value;
  };

  const calculatePoints = () => {
    let total = 0;
    requirements.forEach(req => {
      if (req.id?.startsWith('2-') && req.status === 'required') {
        if (req.value.includes('|')) {
          const [, ptsStr] = req.value.split('|');
          total += parseInt(ptsStr, 10) || 0;
        } else {
          // fallback for legacy cached format like 'B1 (+2分)'
          const match = req.value.match(/\+(\d+)/);
          if (match) total += parseInt(match[1], 10);
          else {
            const fallbackMatch = req.value.match(/(\d+)\s*(?:分|pts)/i);
            if (fallbackMatch) total += parseInt(fallbackMatch[1], 10);
          }
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

    const getFinancialSub = (id: string) => {
      const req = requirements.find(r => r.id === id);
      if (!req) return `0% ${t.met}`;
      if (req.status === 'required') return `100% ${t.met}`;
      const upperVal = req.value.toUpperCase();
      if (upperVal.includes('LACK_OF_FUNDS')) {
        return translateRequirementValue(req.value);
      }
      return `0% ${t.met}`;
    };

    if (cat === 'skilledWorker' || cat === 'work_visa') {
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
      cards.push({ label: t.criteria.financialGoal || 'Financial Requirement', value: `${t.criteria.yearlyAmount}: €11,904`, sub: `(${getFinancialSub('1')})` });
    } else if (cat === 'chancenkarte') {
      // Smart Inference: if points are awarded, base requirements are logically met
      const hasLangPoints = requirements.some(r => r.id === '2-1' && r.status === 'required');
      const hasQualPoints = requirements.some(r => r.id?.match(/^2-[45]/) && r.status === 'required');

      const isLangMet = hasLangPoints || requirements.some(r => r.id === '1-2' && r.status === 'required');
      const isQualMet = hasQualPoints || requirements.some(r => r.id === '1-3' && r.status === 'required');
      const isDirectEligible = requirements.some(r => r.id === '1-3' && r.status === 'required' && r.value !== 'PARTIAL_RECOGNITION');

      const pts = calculatePoints();
      const pointsGoal = 6;
      const outcomeProg = isDirectEligible ? 1 : Math.min(pts / pointsGoal, 1);

      // Total progress is (Language Thresh + Qual Thresh + Points/Direct Goal) / 3
      const overallProg = Math.round((( (isLangMet ? 1 : 0) + (isQualMet ? 1 : 0) + outcomeProg ) / 3) * 100);

      cards.push({
        label: t.criteria.eligibilityPath || 'Eligibility Path',
        value: `${t.criteria.pathDirect}\n${t.criteria.pathPoints}`,
        sub: `(${overallProg}% ${t.met} / ${pts} ${t.criteria.pts})`
      });
      cards.push({ label: t.criteria.financialGoal || 'Financial Requirement', value: `${t.criteria.yearlyAmount}: €13,092`, sub: `(${getFinancialSub('1-1')})` });
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

  const displayChecklist = checklist.map((item) => {
    const displayItem = { ...item };
    // Smart Inference for Chancenkarte Milestones
    if (activeVisaCategory === 'chancenkarte') {
      const hasLangPoints = requirements.some(r => r.id === '2-1' && r.status === 'required');
      const hasQualPoints = requirements.some(r => r.id?.match(/^2-[45]/) && r.status === 'required');
      const isLangMet = hasLangPoints || requirements.some(r => r.id === '1-2' && r.status === 'required');
      const isQualMet = hasQualPoints || requirements.some(r => r.id === '1-3' && r.status === 'required');
      const isFinMet = requirements.some(r => r.id === '1-1' && r.status === 'required');
      const pts = calculatePoints();

      if (item.id === '1') {
         if (isFinMet && isLangMet && isQualMet) displayItem.status = 'completed';
         else if (isFinMet || isLangMet || isQualMet) displayItem.status = 'current';
      }
      if (item.id === '2') {
         if (pts >= 6) displayItem.status = 'completed';
         else if (pts > 0) displayItem.status = 'current';
      }
      if (item.id === '3') {
         if (isFinMet && isLangMet && isQualMet && pts >= 6) displayItem.status = 'current';
      }
    } else if (activeVisaCategory === 'blueCard' || activeVisaCategory === 'blue_card') {
      const isQualMet = requirements.some(r => r.id === '1' && r.status === 'required');
      const isContractMet = requirements.some(r => r.id === '2' && r.status === 'required');
      if (item.id === '1') {
         if (isQualMet) displayItem.status = 'completed';
         else displayItem.status = 'current';
      }
      if (item.id === '2') {
        if (isContractMet) displayItem.status = 'completed';
        else if (requirements.some(r => r.id === '2' && r.status === 'warning') || isQualMet) displayItem.status = 'current';
      }
      if (item.id === '3') {
         if (isQualMet && isContractMet) displayItem.status = 'current';
      }
    } else if (activeVisaCategory === 'studyVisa' || activeVisaCategory === 'student_visa') {
      const isFinMet = requirements.some(r => r.id === '1' && r.status === 'required');
      const isLangMet = requirements.some(r => r.id === '2' && r.status === 'required');
      const isAdmitted = requirements.some(r => r.id === '4' && r.status === 'required');
      if (item.id === '1') {
         if (isAdmitted && isLangMet) displayItem.status = 'completed';
         else displayItem.status = 'current';
      }
      if (item.id === '2') {
        if (isFinMet) displayItem.status = 'completed';
        else if (requirements.some(r => r.id === '1' && r.status === 'warning') || (isAdmitted && isLangMet)) displayItem.status = 'current';
      }
      if (item.id === '3') {
        if (isAdmitted && isLangMet && isFinMet) displayItem.status = 'current';
      }
    } else if (activeVisaCategory === 'skilledWorker' || activeVisaCategory === 'work_visa') {
      const isQualMet = requirements.some(r => r.id === '1' && r.status === 'required');
      const isLaborMet = requirements.some(r => r.id === '2' && r.status === 'required');
      if (item.id === '1') {
         if (isQualMet) displayItem.status = 'completed';
         else displayItem.status = 'current';
      }
      if (item.id === '2') {
        if (isLaborMet) displayItem.status = 'completed';
        else if (requirements.some(r => r.id === '2' && r.status === 'warning') || isQualMet) displayItem.status = 'current';
      }
      if (item.id === '3') {
         if (isQualMet && isLaborMet) displayItem.status = 'current';
      }
    }
    return displayItem;
  });

  return (
    <div className={`glass-panel flex flex-col gap-6 h-full overflow-y-auto p-4 sm:p-5 custom-scrollbar ${className}`}>
      {/* Progress Checklist */}
      <div className="relative">
        <div className="flex items-center justify-between mb-3 lg:mb-4">
          <h3 className="text-base lg:text-[15px] font-semibold text-slate-800 dark:text-slate-200">
            {titlePrefix} {t.progress}
          </h3>
        </div>
        <div className="space-y-4">
          <AnimatePresence mode="popLayout">
            {displayChecklist.map((displayItem, index) => (
              <motion.div
                key={displayItem.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center gap-3 group"
              >
                <div className={`shrink-0 ${displayItem.status === 'completed' ? 'text-green-500' : displayItem.status === 'current' ? 'text-accent' : 'text-slate-300 dark:text-slate-600'}`}>
                  {displayItem.status === 'completed' ? <CheckCircle2 size={18} /> : displayItem.status === 'current' ? (
                    <div className="relative">
                      <Circle size={18} className="animate-pulse" />
                      <div className="absolute inset-0 m-auto w-1.5 h-1.5 bg-accent rounded-full" />
                    </div>
                  ) : <Circle size={18} />}
                </div>
                <div className="flex flex-col min-w-0">
                  <span className={`text-base lg:text-sm font-medium truncate ${displayItem.status === 'completed' ? 'text-slate-400 line-through' : displayItem.status === 'current' ? 'text-slate-800 dark:text-slate-100' : 'text-slate-500'}`}>
                    {translateChecklistTitle(displayItem.title)}
                  </span>
                  <div className="flex items-center gap-2">
                    {displayItem.status === 'current' && displayItem.subtitle && (
                      <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-[10px] text-accent font-medium uppercase tracking-wider"
                      >
                        {displayItem.subtitle}
                      </motion.span>
                    )}
                    {displayItem.autoDetected && (
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
      <div className="mt-2">
        <h3 className="text-base lg:text-[15px] font-semibold text-slate-800 dark:text-slate-200 mb-3 lg:mb-4">{t.reqSummary}</h3>

        <div className="space-y-4">
          {primaryThresholds.map((threshold, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="p-4 bg-slate-50/80 dark:bg-slate-800/40 rounded-xl border border-slate-200/50 dark:border-slate-700/50"
            >
              <div className="flex flex-col gap-2.5">
                <div className="text-xs text-slate-500 dark:text-slate-400 mb-0.5">{threshold.label}:</div>
                <div className="space-y-2">
                  {threshold.value.split('\n').filter(Boolean).map((line: string, i: number) => (
                    <div key={i} className="flex items-start gap-2">
                      <div className="w-1 h-1 rounded-full bg-accent/40 mt-2 shrink-0" />
                      <span className="text-sm lg:text-[13px] font-bold text-slate-800 dark:text-slate-100 leading-tight">
                        {line}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="flex items-center gap-2 mt-4">
                  <div className="text-[10px] font-semibold text-accent bg-accent/10 px-2 py-0.5 rounded-md uppercase tracking-wide">
                    {threshold.sub}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}

          <div className="mt-8">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-3">{t.mainCriteria}:</div>
            <ul className="space-y-3 text-sm text-slate-700 dark:text-slate-300">
              <AnimatePresence mode="popLayout">
                {requirements.map((rawReq, index) => {
                  const req = { ...rawReq };
                  // Apply smart inference to visually satisfy prerequisites if points are awarded
                  if (activeVisaCategory === 'chancenkarte') {
                    const hasLangPoints = requirements.some(r => r.id === '2-1' && r.status === 'required');
                    const hasQualPoints = requirements.some(r => r.id?.match(/^2-[45]/) && r.status === 'required');

                    // A. Forward Inference: Points -> Threshold
                    if (req.id === '1-2' && hasLangPoints && req.status !== 'required') {
                      req.status = 'required';
                      req.value = 'MET';
                    }
                    if (req.id === '1-3' && hasQualPoints && req.status !== 'required') {
                      req.status = 'required';
                      req.value = 'MET';
                    }

                    // B. Reverse Inference: Threshold -> Points (Language)
                    if (req.id === '2-1' && req.status !== 'required') {
                      const tReq = requirements.find(r => r.id === '1-2' && r.status === 'required');
                      if (tReq) {
                        const val = tReq.value.toLowerCase();
                        const ptsMap: Record<string, number> = { a2: 1, b1: 2, b2: 3, c1: 4, en_c1: 1 };
                        if (ptsMap[val]) {
                          req.status = 'required';
                          req.value = `${tReq.value}|${ptsMap[val]}`;
                        }
                      }
                    }

                    // C. Reverse Inference: Threshold -> Points (Qualification)
                    if (req.id === '2-4' && req.status !== 'required') {
                       const tQual = requirements.find(r => r.id === '1-3' && r.status === 'required');
                       if (tQual && tQual.value === 'PARTIAL_RECOGNITION') {
                          req.status = 'required';
                          req.value = 'PARTIAL_RECOGNITION|4';
                       }
                    }

                    // D. Milestone-to-Requirement Inference
                    const isM1Met = checklist.find(i => i.id === '1' && i.status === 'completed');
                    if (isM1Met && req.id?.startsWith('1-')) {
                        req.status = 'required';
                        if (req.value === '-' || !req.value) req.value = 'MET';
                    }
                  }

                  return (
                  <motion.li
                    key={req.id || index}
                    initial={{ opacity: 0, x: 5 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className={`flex justify-between items-start gap-4 p-2 rounded-lg transition-colors group ${req.id?.startsWith('header') ? 'mt-4 mb-1 bg-slate-100/50 dark:bg-slate-800/50 border-l-2 border-indigo-500' : 'hover:bg-slate-50/50 dark:hover:bg-slate-800/30'
                      }`}
                  >
                    <span className={`flex-1 leading-normal ${req.id?.startsWith('header') ? 'text-[11px] font-bold uppercase tracking-wider text-indigo-600 dark:text-indigo-400' : 'text-sm lg:text-xs lg:dark:text-slate-400'
                      }`}>
                      {translateRequirementLabel(req.label, req.id)}
                    </span>
                    {!req.id?.startsWith('header') && (
                      <div className="flex items-center gap-2 pt-[2px] shrink-0">
                        {req.status === 'warning' && <AlertCircle size={14} className="text-amber-500" />}
                          <span className={`min-w-[1.25rem] text-center px-2 py-0.5 rounded-md text-xs lg:text-[11px] font-medium ${req.status === 'required' ? 'bg-green-100 dark:bg-green-500/10 text-green-600 dark:text-green-400' :
                           req.status === 'warning' ? 'bg-amber-100 dark:bg-amber-500/10 text-amber-600 dark:text-amber-400' :
                             'bg-slate-100 dark:bg-slate-800 text-slate-500'
                           }`}>
                          {translateRequirementValue(req.value)}
                        </span>
                      </div>
                    )}
                  </motion.li>
                )})}
              </AnimatePresence>
            </ul>
          </div>
        </div>

        {/* Dynamic Document Checklist View */}
        {displayChecklist.find(c => c.id === '3')?.status !== 'pending' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 pt-5 border-t border-slate-200/50 dark:border-slate-700/50"
          >
            <div className="flex items-center gap-2 mb-5">
              <div className="p-1.5 bg-indigo-500/10 rounded-lg text-indigo-500">
                <FileText size={16} />
              </div>
              <h3 className="text-[13px] lg:text-sm font-bold text-slate-800 dark:text-slate-200">
                {t.docs?.finalTitle || 'Final Document Checklist'}
              </h3>
            </div>

            <div className="space-y-5">
              {/* Common Docs */}
              <div className="p-4 bg-slate-50/80 dark:bg-slate-800/40 rounded-xl border border-slate-200/50 dark:border-slate-700/50 shadow-sm">
                <div className="text-xs font-bold text-slate-500 dark:text-slate-400 mb-3 uppercase tracking-wider flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-500/70"></div>
                  {t.docs?.commonTitle}
                </div>
                <ul className="space-y-1">
                  {[
                    t.docs?.common?.passport,
                    t.docs?.common?.photo,
                    t.docs?.common?.videx,
                    t.docs?.common?.insurance
                  ].map((doc, i) => (
                    <li key={i} className="flex items-start gap-3 p-2.5 rounded-lg hover:bg-white/60 dark:hover:bg-slate-700/40 transition-colors group">
                      <CheckCircle2 size={16} className="text-emerald-500/80 mt-0.5 shrink-0 group-hover:text-emerald-500 transition-colors" />
                      <span className="text-[13px] leading-relaxed text-slate-700 dark:text-slate-300 font-medium">{doc}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Specific Docs */}
              <div className="p-4 bg-indigo-50/50 dark:bg-indigo-900/10 rounded-xl border border-indigo-100 dark:border-indigo-500/20 shadow-sm relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/5 rounded-full blur-3xl -mr-10 -mt-10"></div>
                <div className="text-xs font-bold text-indigo-600/80 dark:text-indigo-400/80 mb-3 uppercase tracking-wider flex items-center gap-2 relative z-10">
                  <div className="w-1.5 h-1.5 rounded-full bg-indigo-500/70"></div>
                  {t.docs?.specificTitle}
                </div>
                <ul className="space-y-1 relative z-10">
                  {(activeVisaCategory === 'student_visa' ? [
                    t.docs?.specific?.studentZulassung,
                    t.docs?.specific?.studentFin,
                    t.docs?.specific?.studentDegree,
                    t.docs?.specific?.studentLang
                  ] : activeVisaCategory === 'chancenkarte' ? [
                    t.docs?.specific?.chancenFin,
                    t.docs?.specific?.chancenQual,
                    t.docs?.specific?.chancenZab,
                    t.docs?.specific?.chancenLangPts,
                    t.docs?.specific?.chancenExpPts
                  ] : activeVisaCategory === 'blue_card' ? [
                    t.docs?.specific?.blueContract,
                    t.docs?.specific?.blueEmp,
                    t.docs?.specific?.blueDegree,
                    t.docs?.specific?.blueIt
                  ] : [
                    t.docs?.specific?.workContract,
                    t.docs?.specific?.workEmp,
                    t.docs?.specific?.workQual,
                    t.docs?.specific?.workPreApp,
                    t.docs?.specific?.workPension
                  ]).map((doc, i) => (
                    <li key={i} className="flex items-start gap-3 p-2.5 rounded-lg hover:bg-white/60 dark:hover:bg-indigo-800/30 transition-colors group">
                      <CheckSquare size={16} className="text-indigo-500/80 mt-0.5 shrink-0 group-hover:text-indigo-500 transition-colors" />
                      <span className="text-[13px] leading-relaxed text-slate-700 dark:text-slate-300 font-medium">{doc}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </motion.div>
        )}

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
