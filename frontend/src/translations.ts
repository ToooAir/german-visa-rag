import { useSettingsStore, Language } from './stores/settingsStore';

export const translations = {
  en: {
    // Sidebar
    home: 'Home',
    library: 'Knowledge Base',
    settings: 'Settings',
    recentSources: 'Recent Sources',
    visaCategories: 'Visa Categories',
    
    // Settings
    appearance: 'Appearance',
    themePreference: 'Theme Preference',
    langRegion: 'Language & Region',
    chatLang: 'Chat Language',
    notifications: 'Notifications',
    emailAlerts: 'Email Alerts for Visa Updates',
    smartAlerts: 'Smart Progress Alerts',
    smartAlertsDesc: 'Get notified when AI updates your milestone.',
    milestoneReached: 'Milestone Reached!',
    milestoneAutoUpdated: 'AI has detected progress and automatically updated your checklist.',
    dark: 'Dark Mode',
    light: 'Light Mode',
    system: 'System Default',
    
    // Chat
    welcome: 'Hello! I am **VisaPilot AI**. I can help you understand German visa regulations, the *Chancenkarte*, and official requirements.',
    chatHeader: 'Chat',
    askPlaceholder: 'Ask VisaPilot AI...',
    showcaseExamples: 'Showcase Examples',
    you: 'You',
    disclaimer: 'This response is based on public information and for reference only; it does not constitute legal advice. Please consult official sources or professionals for critical decisions.',
    
    // Insights
    progress: 'Progress',
    reqSummary: 'Requirement Summary',
    pointsReq: 'Points Required',
    mainCriteria: 'Main Criteria',
    met: 'met',
    analyzing: 'Analyzing intent & parameters...',
    retrieving: 'Querying official vector database...',
    extracting: 'Extracting legal requirements...',
    synthesizing: 'Synthesizing answer...',
    docChecklist: 'Document Checklist',
    embassyAppt: 'Embassy Appointment',
    approval: 'Approval',
    salaryCheck: 'Salary Check',
    eligibilityDesc: 'Based on your current profile, you meet the primary criteria but need a few more points for the Chancenkarte.',
    resetProgress: 'Reset Progress',
    autoDetected: 'Auto-detected by AI',
    autoDetectedDesc: 'This milestone was automatically identified from your conversation history.',
    
    // Visa Categories
    chancenkarte: 'Chancenkarte',
    skilledWorker: 'Skilled Worker',
    blueCard: 'Blue Card',
    studyVisa: 'Study Visa',

    // Showcase
    showcase: {
      pointCalc: { title: 'Point Calculation', query: 'Am I eligible for the Chancenkarte with 5 years experience and B1 German?' },
      blueCardSalary: { title: 'Blue Card Salary', query: 'What is the new minimum salary for the EU Blue Card in 2026?' },
      studentWork: { title: 'Student Work Rights', query: 'Can international students work part-time? Explain the latest updates.' }
    },

    // Requirements
    criteria: {
      language: 'Language Proficiency',
      experience: 'Work Experience',
      age: 'Age',
      qualifications: 'Academic & Professional Qualifications',
      eligibility: 'Eligibility Check',
      salaryThreshold: 'Salary Threshold',
      blockedAccount: 'Blocked Account',
      financialProof: 'Financial Proof',
      coreRequirement: 'Core Requirement',
      pathDirect: 'Qualification Path (Direct)',
      pathPoints: 'Score-based Path (6+ Points)',
      eligibilityPath: 'Eligibility Path',
      recognitionPartnership: 'Recognition Partnership (FEG 2.0)',
      fullTimeContract: 'German Full-time Contract',
      blueCardGen: 'Standard',
      blueCardIT: 'Shortage / IT & STEM',
      laborConditions: 'Labor Conditions',
      age45Clause: '45+ Age Clause',
      flexibleLanguage: 'Language (Flexible)',
      contractDuration: 'Contract Duration (min 6mo)',
      itExemption: 'IT Experience Exemption (3yr)',
      prLanguageBonus: 'PR Fast-track (B1 = 21mo, Optional)',
      uniAdmission: 'University Admission (Zulassung)',
      healthInsurance: 'Health Insurance',
      scholarship: 'Scholarship Proof',
      commitmentLetter: 'Commitment Letter (VE)',
      preStudyQuals: 'Pre-study Qualifications',
      residencyBonus: 'German Residency (6mo+)',
      partnerBonus: 'Partner Bonus',
      basicThresholds: 'Mandatory Thresholds',
      pointsItems: 'Points Items (Target 6+)',
      pts: 'pts',
      langPoints: 'Language Points',
      expPoints: 'Experience Points',
      agePoints: 'Age Points',
      qualPoints: 'Quals/Shortage Points',
      altFinancial: 'Alternative Financial Proofs',
      salaryStandards: 'Salary commensurate with standards (BA check)',
      financialGoal: 'Subsistence Funds (Blocked Acc.)',
      yearlyAmount: 'Total / year'
    }
  },
  de: {
    // Sidebar
    home: 'Startseite',
    library: 'Wissensdatenbank',
    settings: 'Einstellungen',
    recentSources: 'Aktuelle Quellen',
    visaCategories: 'Visum-Kategorien',
    
    // Settings
    appearance: 'Erscheinungsbild',
    themePreference: 'Design-Einstellung',
    langRegion: 'Sprache & Region',
    chatLang: 'Chat-Sprache',
    notifications: 'Benachrichtigungen',
    emailAlerts: 'E-Mail-Benachrichtigungen für Visum-Updates',
    smartAlerts: 'Intelligente Fortschrittsalarme',
    smartAlertsDesc: 'Werden Sie benachrichtigt, wenn die KI Ihren Meilenstein aktualisiert.',
    milestoneReached: 'Meilenstein Erreicht!',
    milestoneAutoUpdated: 'KI hat Ihre Fortschritts-Checkliste automatisch aktualisiert.',
    dark: 'Dunkler Modus',
    light: 'Heller Modus',
    system: 'Systemstandard',
    
    // Chat
    welcome: 'Hallo! Ich bin **VisaPilot AI**. Ich kann Ihnen helfen, deutsche Visumbestimmungen, die *Chancenkarte* und offizielle Anforderungen zu verstehen.',
    chatHeader: 'Chat',
    askPlaceholder: 'Fragen Sie VisaPilot AI...',
    showcaseExamples: 'Beispiele ansehen',
    you: 'Sie',
    disclaimer: 'Diese Antwort basiert auf öffentlichen Informationen und dient nur zu Referenzzwecken; sie stellt keine Rechtsberatung dar. Bitte konsultieren Sie offizielle Quellen für wichtige Entscheidungen.',
    
    // Insights
    progress: 'Fortschritt',
    reqSummary: 'Anforderungszusammenfassung',
    pointsReq: 'Erforderliche Punkte',
    mainCriteria: 'Hauptkriterien',
    met: 'erfüllt',
    analyzing: 'Intent & Parameter werden analysiert...',
    retrieving: 'Offizielle Vektordatenbank wird abgefragt...',
    extracting: 'Rechtliche Anforderungen werden extrahiert...',
    synthesizing: 'Antwort wird synthetisiert...',
    docChecklist: 'Dokumenten-Checkliste',
    embassyAppt: 'Botschaftstermin',
    approval: 'Genehmigung',
    salaryCheck: 'Gehaltsprüfung',
    eligibilityDesc: 'Basierend auf Ihrem aktuellen Profil erfüllen Sie die Hauptkriterien, benötigen aber noch einige Punkte für die Chancenkarte.',
    resetProgress: 'Fortschritt zurücksetzen',
    autoDetected: 'KI-automatisch erkannt',
    autoDetectedDesc: 'Dieser Meilenstein wurde automatisch aus Ihrem Gesprächsverlauf identifiziert.',

    // Visa Categories
    chancenkarte: 'Chancenkarte',
    skilledWorker: 'Fachkraft',
    blueCard: 'Blaue Karte EU',
    studyVisa: 'Visum zum Studium',

    // Showcase
    showcase: {
      pointCalc: { title: 'Punkteberechnung', query: 'Bin ich mit 5 Jahren Erfahrung und B1 Deutsch für die Chancenkarte berechtigt?' },
      blueCardSalary: { title: 'Gehalt Blaue Karte', query: 'Was ist das neue Mindestgehalt für die Blaue Karte EU im Jahr 2026?' },
      studentWork: { title: 'Arbeitsrechte für Studenten', query: 'Können internationale Studierende Teilzeit arbeiten? Erklären Sie die neuesten Updates.' }
    },

    // Requirements
    criteria: {
      language: 'Sprachkenntnisse',
      experience: 'Berufserfahrung',
      age: 'Alter',
      qualifications: 'Akademische & Berufliche Qualifikationen',
      eligibility: 'Berechtigungsprüfung',
      salaryThreshold: 'Gehaltsschwelle',
      blockedAccount: 'Sperrkonto',
      financialProof: 'Finanzierungsnachweis',
      coreRequirement: 'Kernanforderung',
      pathDirect: 'Anerkennungsweg (Direkt)',
      pathPoints: 'Punktesystem (6+ Pkt.)',
      eligibilityPath: 'Berechtigungspfad',
      recognitionPartnership: 'Anerkennungspartnerschaft (FEG 2.0)',
      fullTimeContract: 'Deutscher Vollzeit-Arbeitsvertrag',
      blueCardGen: 'Regelberufe',
      blueCardIT: 'Mangelberufe / IT & Technik',
      laborConditions: 'Arbeitsbedingungen',
      age45Clause: '45+ Klausel',
      flexibleLanguage: 'Sprache (Flexibel)',
      contractDuration: 'Vertragsdauer (min 6 Mon.)',
      itExemption: 'IT-Erfahrung (3 J., ohne Abschluss)',
      prLanguageBonus: 'PR-Vorteil (B1 = 21 Mon., Optional)',
      uniAdmission: 'Zulassungsbescheid',
      healthInsurance: 'Krankenversicherung',
      scholarship: 'Stipendiennachweis',
      commitmentLetter: 'Verpflichtungserklärung (VE)',
      preStudyQuals: 'Vorstudienqualifikationen',
      residencyBonus: 'Voraufenthalt in DE (min. 6 Mon.)',
      partnerBonus: 'Partner-Bonus',
      basicThresholds: 'Basisanforderungen',
      pointsItems: 'Punktekategorien (Ziel 6+)',
      pts: 'Pkt.',
      langPoints: 'Sprachpunkte',
      expPoints: 'Erfahrungspunkte',
      agePoints: 'Alterspunkte',
      qualPoints: 'Qualifikationspunkte',
      altFinancial: 'Alternative Finanzierungsnachweise',
      salaryStandards: 'Gehalt ortsüblich & branchentypisch (BA)',
      financialGoal: 'Lebensunterhalt (Sperrkonto)',
      yearlyAmount: 'Gesamt / Jahr'
    }
  },
  'zh-TW': {
    // Sidebar
    home: '首頁',
    library: '法規知識庫',
    settings: '設定',
    recentSources: '最近來源',
    visaCategories: '簽證類別',
    
    // Settings
    appearance: '外觀',
    themePreference: '主題偏好',
    langRegion: '語言與區域',
    chatLang: '聊天語言',
    notifications: '通知',
    emailAlerts: '簽證更新電子郵件通知',
    smartAlerts: '智能進度提醒',
    smartAlertsDesc: '當 AI 自動更新您的里程碑時接收通知',
    milestoneReached: '🎊 里程碑達成！',
    milestoneAutoUpdated: 'AI 已偵測進度並自動更新您的檢核表。',
    dark: '深色模式',
    light: '淺色模式',
    system: '系統預設',
    
    // Chat
    welcome: '您好！我是 **VisaPilot AI**。我可以協助您了解德國簽證法規、*機會卡 (Chancenkarte)* 以及官方要求。',
    chatHeader: '聊天',
    askPlaceholder: '詢問 VisaPilot AI...',
    showcaseExamples: '展示範例',
    you: '您',
    disclaimer: '本回答內容僅供參考，不構成法律建議。所有重要簽證決定請務必諮詢官方機構或專業法律人士。',
    
    // Insights
    progress: '當前進度',
    reqSummary: '要求摘要',
    pointsReq: '積分門檻',
    mainCriteria: '主要標準',
    met: '達成',
    analyzing: '正在規劃搜尋參數...',
    retrieving: '正在檢索官方法規資料庫...',
    extracting: '正在提取相關條款與細節...',
    synthesizing: '正在彙整簽證建議與回答...',
    docChecklist: '文件清單',
    embassyAppt: '使館面談預約',
    approval: '核准',
    salaryCheck: '薪資門檻查驗',
    eligibilityDesc: '根據您目前的個人資料，您已符合主要標準，但還需要更多積分才能獲得機會卡。',
    resetProgress: '重置進度',
    autoDetected: 'AI 自動偵測',
    autoDetectedDesc: '此進度是根據您的對話紀錄自動分析得出的。',

    // Visa Categories
    chancenkarte: '機會卡',
    skilledWorker: '技術人才簽證',
    blueCard: '藍卡 (Blue Card)',
    studyVisa: '就學簽證',

    // Showcase
    showcase: {
      pointCalc: { title: '積分計算', query: '我有 5 年經驗且德語 B1，符合申請機會卡的資格嗎？' },
      blueCardSalary: { title: '藍卡薪資門檻', query: '2026 年歐盟藍卡的新最低薪資要求是多少？' },
      studentWork: { title: '學生工作權利', query: '外籍學生可以兼職工作嗎？請解釋最新的規定。' }
    },

    // Requirements
    criteria: {
      language: '語言能力',
      experience: '工作經驗',
      age: '年齡',
      qualifications: '學歷與專業資格',
      eligibility: '資格審查',
      salaryThreshold: '年薪門檻',
      blockedAccount: '限制提領帳戶',
      financialProof: '財力證明',
      coreRequirement: '核心條件',
      pathDirect: '專業資格 (直接認可)',
      pathPoints: '積分計分路徑',
      eligibilityPath: '資格路徑',
      recognitionPartnership: '認可夥伴關係 (FEG 2.0)',
      fullTimeContract: '德國全職工作合約 (或具體聘僱意向書)',
      blueCardGen: '一般職業 / 標準',
      blueCardIT: '稀缺職業 / IT・專業人士',
      laborConditions: '薪資與勞動條件',
      age45Clause: '45歲以上特殊條款',
      flexibleLanguage: '語言能力 (彈性控制)',
      contractDuration: '合約效期 (至少 6 個月)',
      itExemption: 'IT 人才專屬豁免 (3年經驗)',
      prLanguageBonus: '永居縮短 (B1 滿 21 個月，非必備)',
      uniAdmission: '德國大學錄取通知書 (Zulassung)',
      healthInsurance: '德國醫療保險 (必備)',
      scholarship: '獎學金證明',
      commitmentLetter: '經濟擔保書 (VE)',
      preStudyQuals: '前置學經歷資格 (Uni-assist)',
      residencyBonus: '德國居住經歷加分 (滿 6 個月)',
      partnerBonus: '伴侶共同申請加分',
      basicThresholds: '必備基本門檻',
      pointsItems: '積分項目 (需獲 6 分以上)',
      pts: '分',
      langPoints: '語言加分',
      expPoints: '工作經驗加分',
      agePoints: '年齡加分',
      qualPoints: '學經歷/稀缺加分',
      altFinancial: '其他財力證明方式',
      salaryStandards: '薪資需達當地同業水準 (由勞動局 BA 審核)',
      financialGoal: '生活費證明 (限制提領帳戶)',
      yearlyAmount: '年總額'
    }
  }
};

export const useTranslation = () => {
  const language = useSettingsStore((state) => state.language);
  const t = translations[language as Language] || translations.en;
  return { t, language };
};
