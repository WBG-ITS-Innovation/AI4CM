# -*- coding: utf-8 -*-
"""Georgian translations. THIS IS THE ONE FILE A NATIVE SPEAKER NEEDS TO CORRECT.

═══════════════════════════════════════════════════════════════════════════════
STATUS: MACHINE GENERATED, NOT REVIEWED BY A NATIVE SPEAKER
═══════════════════════════════════════════════════════════════════════════════

Every Georgian string below was produced by a language model. None of it has been checked
by a Georgian speaker. It is a starting point for that review and nothing more, and the
app says so on every page while Georgian is selected.

How to correct it
-----------------
Each entry is one English sentence and its Georgian. Edit the right-hand side. There is no
key to look up, no other file to touch, and nothing to rebuild: save this file and reload
the page.

    "Start here": "დაიწყე აქ",
     ^ the English exactly as it appears in the app
                  ^ the Georgian, which is yours to correct

Two rules for a corrector
-------------------------
1. **Leave the left-hand side alone.** It is the lookup key and must match the English in
   the app character for character, including punctuation. Changing it silently disables
   the translation rather than causing an error.
2. **Keep any ``{}`` where it is.** A figure or a name is substituted there at display
   time. Georgian word order may want it in a different position in the sentence, which is
   fine; what matters is that it is still present.

What is deliberately NOT translated
-----------------------------------
Treasury line names, model names, dates, figures, file paths and column headings. A line
called "State budget balance" is named that in the source data, and translating it would
break the correspondence a reader needs in order to check a number against the file it
came from.

Coverage
--------
Partial by design rather than by neglect. The short, high-traffic strings that carry the
meaning are here: the interface chrome, every page's opening sentences, the glossary, the
verdict wording, and the guide page's structure. The longest methodological passages are
not, because a machine translation of a subtle paragraph is worse than an untranslated one
a reader can still read in English. ``i18n.coverage()`` measures what is present and the
language note on every page states the figure, so the gap is visible rather than implied.
"""

TRANSLATIONS = {
    # ── Interface chrome ────────────────────────────────────────────────────
    "Start here": "დაიწყე აქ",
    "Forward forecast": "სამომავლო პროგნოზი",
    "Scorecard": "შედეგების ბარათი",
    "Dashboard": "მაჩვენებლების დაფა",
    "Compare runs": "გაშვებების შედარება",
    "Run history": "გაშვებების ისტორია",
    "Documentation": "დოკუმენტაცია",
    "Georgian State Treasury Forecast Lab":
        "საქართველოს სახელმწიფო ხაზინის პროგნოზის ლაბორატორია",
    "Data pre-processing": "მონაცემთა წინასწარი დამუშავება",
    "Lab": "ლაბორატორია",
    "Overview": "მიმოხილვა",
    "Forecast": "პროგნოზი",
    "How to read this page": "როგორ წავიკითხოთ ეს გვერდი",
    "Two kinds of forecast": "პროგნოზის ორი სახე",
    "Official": "ოფიციალური",
    "Exploratory": "საძიებო",

    # ── Copy added by the tabs, the scorecard loop and the Lab downloads ────
    #
    # Short structural copy: headings, table cells, labels and lead-ins. The long
    # methodological passages this session added are deliberately NOT here, per the
    # policy in this file's docstring: a machine translation of a subtle paragraph is
    # worse than an untranslated one a reader can still read in English.
    "passed": "გაიარა",
    "failed": "ვერ გაიარა",
    "Results": "შედეგები",
    "Yes": "დიახ",
    "No": "არა",
    "None": "არცერთი",
    "Low (P10)": "დაბალი (P10)",
    "Central (P50)": "ცენტრალური (P50)",
    "High (P90)": "მაღალი (P90)",
    "What it is": "რა არის ეს",
    "For the day": "რომელი დღისთვის",
    "Issued": "გამოცემულია",
    "Treasury line": "ხაზინის ხაზი",
    "Champion model": "ჩემპიონი მოდელი",
    "Upload actuals": "ფაქტობრივი მონაცემების ატვირთვა",
    "Checks measured": "გაზომილი შემოწმებები",
    "Can be published": "შეიძლება გამოქვეყნდეს",
    "Scored against reality": "შედარებულია რეალობასთან",
    "Who picks the model": "ვინ ირჩევს მოდელს",
    "You do, from a list": "თქვენ, სიიდან",
    "Published forecasts": "გამოქვეყნებული პროგნოზები",
    "Exploratory runs.": "საძიებო გაშვებები.",
    "Read the procedure": "წაიკითხეთ პროცედურა",
    "How the loop works.": "როგორ მუშაობს ციკლი.",
    "The risk this leaves.": "რა რისკს ტოვებს ეს.",
    "What published means.": "რას ნიშნავს გამოქვეყნებული.",
    "Some lines are withheld.": "ზოგიერთი ხაზი შეკავებულია.",
    "When you upload actuals.": "როდესაც ატვირთავთ ფაქტობრივ მონაცემებს.",
    "Waiting for actual figures": "ელოდება ფაქტობრივ მაჩვენებლებს",
    "Already scored": "უკვე შეფასებული",
    "What never changes on its own.": "რა არ იცვლება თავისით.",
    "All of them, each with a reason": "ყველა, თითოეული მიზეზით",
    "What re-choosing would require.": "რას მოითხოვდა ხელახალი არჩევა.",
    "The forecast for a Treasury line": "პროგნოზი ხაზინის ერთი ხაზისთვის",
    "No, and there is no setting for it":
        "არა, და ამისთვის პარამეტრი არ არსებობს",
    "Nobody here. Its evidence did, once":
        "აქ არავინ. მისმა მტკიცებულებამ, ერთხელ",
    "The sealed window is not read here.": "დალუქული ფანჯარა აქ არ იკითხება.",
    "If the results above say degrading.":
        "თუ ზემოთ მოცემული შედეგები ამბობს, რომ უარესდება.",
    "A published forecast is never edited.":
        "გამოქვეყნებული პროგნოზი არასოდეს რედაქტირდება.",
    "How the scored forecasts actually did":
        "როგორ გამოვიდა შეფასებული პროგნოზები სინამდვილეში",
    "Read the refresh and retrain procedure":
        "წაიკითხეთ განახლებისა და ხელახალი წვრთნის პროცედურა",
    "Why re-choosing is a deliberate decision":
        "რატომ არის ხელახალი არჩევა შეგნებული გადაწყვეტილება",
    "Any model on any line, to see what it does":
        "ნებისმიერი მოდელი ნებისმიერ ხაზზე, რომ ვნახოთ რას აკეთებს",
    "Too few scored days to call it either way.":
        "შეფასებული დღეები ძალიან ცოტაა, რომ რომელიმე მხარეს დავასკვნათ.",
    "The rows that are waiting are listed above.":
        "მოლოდინში მყოფი მწკრივები ზემოთ ჩამოთვლილია.",
    "What happens to the model when new data arrives":
        "რა ხდება მოდელთან, როდესაც ახალი მონაცემები მოდის",
    "It is refitted. It is never re-chosen on its own":
        "ის ხელახლა ეწყობა. ის არასოდეს ირჩევა ხელახლა თავისით",
    "Measured instances the registry already records.":
        "გაზომილი შემთხვევები, რომლებსაც რეესტრი უკვე აღრიცხავს.",
    "None scored yet, so the results section below is empty.":
        "ჯერ არცერთი არ არის შეფასებული, ამიტომ ქვემოთ მოცემული შედეგების ნაწილი ცარიელია.",
    "Nothing has been scored yet, and that is the honest state.":
        "ჯერ არაფერია შეფასებული და ეს არის პატიოსანი მდგომარეობა.",
    "Three things worth knowing before you read the figures below":
        "სამი რამ, რაც ღირს იცოდეთ ქვემოთ მოცემული ციფრების წაკითხვამდე",
    "Every forecast that was issued, and whether it has been scored":
        "ყოველი გამოცემული პროგნოზი და შეფასდა თუ არა ის",
    "Which one you are making is the most useful thing to know here":
        "რომელს აკეთებთ, ეს არის ყველაზე სასარგებლო, რაც აქ უნდა იცოდეთ",
    "Add the days the Treasury has now reported, and score against them":
        "დაამატეთ დღეები, რომლებიც ხაზინამ ახლა დააფიქსირა, და შეაფასეთ მათ მიხედვით",
    "Holding up.": "ინარჩუნებს მდგომარეობას.",
    "Degrading.": "უარესდება.",
    "Only": "მხოლოდ",
    "Over": "განმავლობაში",
    "Take the files with you": "წაიღეთ ფაილები თქვენთან",
    "This run's folder": "ამ გაშვების საქაღალდე",
    "Everything (.zip)": "ყველაფერი (.zip)",
    "Open History": "გახსენით ისტორია",
    "Adding a model": "მოდელის დამატება",
    "What you can do here.": "რა შეგიძლიათ აქ გააკეთოთ.",
    "Where these numbers come from.": "საიდან მოდის ეს ციფრები.",
    "train and dev": "სასწავლო და განვითარების მონაცემები",
    "horizon h": "ჰორიზონტი h",
    "run folder": "გაშვების საქაღალდე",
    "One run folder under frontend/runs/. Nothing here is recomputed.":
        "ერთი გაშვების საქაღალდე frontend/runs/-ში. აქ არაფერი ხელახლა არ გამოითვლება.",
    "The file you upload. Nothing on this page reads the canonical data.":
        "ფაილი, რომელსაც ატვირთავთ. ამ გვერდზე არაფერი კითხულობს კანონიკურ მონაცემებს.",
    "The run folders under frontend/runs/, and the progress record in reports/.":
        "გაშვების საქაღალდეები frontend/runs/-ში და პროგრესის ჩანაწერი reports/-ში.",
    "The run folders you select. Each figure is read from the run that produced it.":
        "თქვენ მიერ არჩეული გაშვების საქაღალდეები. ყოველი ციფრი იკითხება იმ გაშვებიდან, რომელმაც ის შექმნა.",
    "issue(s).": "გამოცემა(ები).",
    "published row(s) across": "გამოქვეყნებული მწკრივ(ებ)ი, სულ",

    # ── Page subtitles ──────────────────────────────────────────────────────
    "What was forecast, and what actually happened":
        "რა იყო ნაპროგნოზები და რა მოხდა სინამდვილეში",
    "Evaluate one run: accuracy, intervals and integrity checks":
        "ერთი გაშვების შეფასება: სიზუსტე, ინტერვალები და მთლიანობის შემოწმებები",
    "Put several runs side by side on the same target and horizon":
        "რამდენიმე გაშვების გვერდიგვერდ განთავსება ერთსა და იმავე მიზანსა და ჰორიზონტზე",
    "Browse past runs and download their outputs":
        "წარსული გაშვებების დათვალიერება და მათი შედეგების ჩამოტვირთვა",
    "Every model, its settings, the promoted recipes and their evidence":
        "ყოველი მოდელი, მისი პარამეტრები, დაწინაურებული რეცეპტები და მათი მტკიცებულებები",
    "Build and inspect the canonical daily Treasury file":
        "ხაზინის კანონიკური დღიური ფაილის შექმნა და შემოწმება",
    "Try any model on any Treasury line, at any horizon, safely":
        "სცადეთ ნებისმიერი მოდელი ხაზინის ნებისმიერ ხაზზე, ნებისმიერ ჰორიზონტზე, უსაფრთხოდ",
    "What this lab does, and what it does not claim":
        "რას აკეთებს ეს ლაბორატორია და რას არ ამტკიცებს",

    # ── Page intros ─────────────────────────────────────────────────────────
    "This page is the way in. It says what each page of the Lab is for, what you can do "
    "there, and the difference between an official forecast and an experiment.":
        "ეს გვერდი შესასვლელია. აქ წერია, რისთვის არის ლაბორატორიის თითოეული გვერდი, "
        "რა შეგიძლიათ იქ გააკეთოთ და რა განსხვავებაა ოფიციალურ პროგნოზსა და "
        "ექსპერიმენტს შორის.",

    "This page holds the forecast itself: what each Treasury line is expected to do over "
    "the next few working days, which model produced each figure, and what earned that "
    "model its place.":
        "ამ გვერდზეა თავად პროგნოზი: რას უნდა ველოდოთ ხაზინის თითოეული ხაზისგან "
        "მომდევნო რამდენიმე სამუშაო დღეში, რომელმა მოდელმა შექმნა თითოეული ციფრი და "
        "რამ დაიმსახურა ამ მოდელისთვის თავისი ადგილი.",

    "This page compares published forecasts to what actually happened, once the day "
    "arrives. It is also where newly reported actuals are uploaded.":
        "ეს გვერდი ადარებს გამოქვეყნებულ პროგნოზებს იმას, რაც სინამდვილეში მოხდა, "
        "როდესაც დღე დგება. აქვე იტვირთება ახლად დაფიქსირებული ფაქტობრივი მონაცემები.",

    "This page shows the detail behind one experimental run: how accurate it was, where "
    "its errors fell, and whether its checks passed. Nothing here is published.":
        "ეს გვერდი აჩვენებს ერთი ექსპერიმენტული გაშვების დეტალებს: რამდენად ზუსტი იყო, "
        "სად დაფიქსირდა მისი შეცდომები და გაიარა თუ არა შემოწმებები. აქედან არაფერი "
        "ქვეყნდება.",

    "This page puts two or more experimental runs side by side on the same axes, so a "
    "difference between them can be seen rather than assumed.":
        "ეს გვერდი ორ ან მეტ ექსპერიმენტულ გაშვებას ათავსებს გვერდიგვერდ ერთსა და იმავე "
        "ღერძებზე, რათა მათ შორის განსხვავება დაინახოთ და არა ივარაუდოთ.",

    "This page lists every experimental run this Lab has produced, including the ones "
    "that failed their checks, and lets you download what each of them wrote.":
        "ეს გვერდი ჩამოთვლის ამ ლაბორატორიის ყველა ექსპერიმენტულ გაშვებას, მათ შორის "
        "იმებს, რომლებმაც შემოწმებები ვერ გაიარეს, და საშუალებას გაძლევთ ჩამოტვირთოთ "
        "თითოეულის შედეგები.",

    "This page is the shelf: every model available here, what it does in plain language, "
    "and whether anybody has recorded a measured result for it.":
        "ეს გვერდი არის თარო: ყველა აქ ხელმისაწვდომი მოდელი, რას აკეთებს იგი მარტივი "
        "ენით და აქვს თუ არა ვინმეს დაფიქსირებული გაზომილი შედეგი მისთვის.",

    "This page turns a raw Treasury export into the clean daily series the models read, "
    "and reports what the cleaning changed so a surprising number can be traced back to "
    "it.":
        "ეს გვერდი ხაზინის დაუმუშავებელ ფაილს გარდაქმნის სუფთა დღიურ მწკრივად, რომელსაც "
        "მოდელები კითხულობენ, და აღწერს რა შეიცვალა გასუფთავებისას, რათა მოულოდნელი "
        "ციფრი უკან იქამდე იქნას მიკვლეული.",

    "Try any model on any Treasury line, at any horizon, safely. Nothing here is published, "
    "nothing here changes the official forecast, and every result is measured on training and "
    "development data only.":
        "სცადეთ ნებისმიერი მოდელი ხაზინის ნებისმიერ ხაზზე, ნებისმიერ ჰორიზონტზე, "
        "უსაფრთხოდ. აქედან არაფერი ქვეყნდება, აქ არაფერი ცვლის ოფიციალურ პროგნოზს "
        "და ყოველი შედეგი იზომება მხოლოდ სასწავლო და განვითარების მონაცემებზე.",

    "This is the landing page. It confirms the Lab can find its backend, shows what has "
    "been run so far, and states plainly what this project does and does not claim.":
        "ეს არის მთავარი გვერდი. იგი ადასტურებს, რომ ლაბორატორია პოულობს თავის სერვერულ "
        "ნაწილს, აჩვენებს რა გაშვებულა აქამდე და ნათლად აცხადებს, რას ამტკიცებს და რას "
        "არ ამტკიცებს ეს პროექტი.",

    # ── The glossary: terms ─────────────────────────────────────────────────
    "champion": "ჩემპიონი",
    "exploratory": "საძიებო",
    "baseline": "საბაზისო წესი",
    "skill": "უპირატესობა",
    "MASE": "MASE",
    "holdout": "გამოყოფილი მონაცემები",
    "sealed window": "დალუქული ფანჯარა",
    "gate": "შემოწმება",
    "withheld": "შეჩერებული",
    "P10": "P10",
    "P50": "P50",
    "P90": "P90",
    "pending": "მოლოდინში",

    # ── The glossary: definitions ───────────────────────────────────────────
    "The one model an official forecast for a given Treasury line uses. It was chosen "
    "once, on recorded evidence from data it had never been fitted on, and loading new "
    "data refits it without ever re-choosing it.":
        "ერთადერთი მოდელი, რომელსაც ხაზინის მოცემული ხაზის ოფიციალური პროგნოზი იყენებს. "
        "იგი შეირჩა ერთხელ, დაფიქსირებული მტკიცებულების საფუძველზე, იმ მონაცემებზე, "
        "რომლებზეც არასოდეს ყოფილა მორგებული. ახალი მონაცემების ჩატვირთვა მას თავიდან "
        "არგებს, მაგრამ არასოდეს ირჩევს ხელახლა.",

    "A run you launched yourself to see what a model would do. It is measured on the "
    "training and development data only, it is never published, it is never written to the "
    "official forecast, and it never enters the scorecard. Every page that produces one "
    "says so while it is showing it.":
        "გაშვება, რომელიც ვინმემ დაიწყო იმის სანახავად, თუ რა მოხდებოდა. იგი არასოდეს "
        "ქვეყნდება, არასოდეს ჩაიწერება ოფიციალურ პროგნოზში და არასოდეს შედის შედეგების "
        "ბარათში. ყოველი გვერდი, რომელიც მას ქმნის, ამას აჩვენებისას აცხადებს.",

    "A deliberately simple rule that every model is measured against, such as assuming "
    "the value from five working days ago simply repeats. Beating it is the floor, not an "
    "achievement.":
        "განზრახ მარტივი წესი, რომელსაც ყველა მოდელი ედარება, მაგალითად ვარაუდი, რომ ხუთი "
        "სამუშაო დღის წინანდელი მნიშვნელობა უბრალოდ მეორდება. მისი დამარცხება მინიმალური "
        "ზღვარია და არა მიღწევა.",

    "How much smaller a model's typical error is than the baseline's, as a percentage. The "
    "baseline holds the last known figure flat, so at this project's horizon of five "
    "working days it is the figure from five working days earlier. 40% means the model's "
    "errors are 40% smaller than that.":
        "რამდენად ნაკლებია მოდელის ტიპიური შეცდომა საბაზისო წესის შეცდომაზე, პროცენტებში. "
        "40% ნიშნავს, რომ მისი შეცდომები 40%-ით ნაკლებია, ვიდრე ბოლო ცნობილი მნიშვნელობის "
        "გამეორების ვარაუდისას.",

    "A model's error divided by the error of repeating the same weekday from the previous "
    "week. Below 1.00 means better than that simple rule and above 1.00 means worse, so "
    "1.00 is the break-even point rather than a threshold anybody chose.":
        "მოდელის შეცდომა გაყოფილი წინა კვირის იმავე კვირის დღის გამეორების შეცდომაზე. "
        "1.00-ზე ნაკლები ნიშნავს, რომ ის ჯობია ამ მარტივ წესს, ხოლო 1.00-ზე მეტი ნიშნავს, "
        "რომ ჩამორჩება. ამიტომ 1.00 არის თანაბრობის წერტილი და არა ვინმეს მიერ არჩეული "
        "ზღვარი.",

    "A block of history deliberately kept away from the models while they were being "
    "chosen, so that measuring them on it says something about days they had never seen.":
        "ისტორიის მონაკვეთი, რომელიც განზრახ იყო დაფარული მოდელებისგან მათი შერჩევისას, "
        "რათა მასზე გაზომვამ რაიმე თქვას იმ დღეებზე, რომლებიც მათ არასოდეს უნახავთ.",

    "The most recent stretch of history the models never saw while being chosen. We keep it "
    "untouched so the final score is honest. It can only be spent once, so no experiment "
    "may be measured on it, and any that tries is refused.":
        "ისტორიის უახლესი მონაკვეთი, რომელიც შენახულია და ბოლოს ერთხელ იკითხება. ეს არის "
        "ამ პროექტის ერთადერთი სუფთა საბოლოო წაკითხვა, ამიტომ არცერთ ექსპერიმენტს არ "
        "აქვს მასზე შეხების უფლება და ყოველი მცდელობა უარყოფილია.",

    "A check a forecast must pass before it may be published, such as being more accurate "
    "than the simple rule of thumb. Every gate has a plain-language reason attached to "
    "its verdict, and none can be switched off from this interface.":
        "შემოწმება, რომელიც პროგნოზმა უნდა გაიაროს გამოქვეყნებამდე, მაგალითად მარტივ "
        "წესზე უფრო ზუსტი უნდა იყოს. ყოველ შემოწმებას თავის დასკვნასთან ერთად ახლავს "
        "მარტივ ენაზე ახსნილი მიზეზი და არცერთის გამორთვა ამ ინტერფეისიდან შეუძლებელია.",

    "Withheld means we do not offer the numbers as a forecast. It happens for one of two "
    "reasons. Either a simple rule of thumb was more accurate, so the numbers should not be "
    "used at all. Or the model could not show that it anticipates individual days, so the "
    "numbers are a guide to the typical level and nothing more. The page always says which "
    "of the two applies.":
        "დასკვნა, რომელიც ნიშნავს, რომ ციფრები არ არის შემოთავაზებული როგორც პროგნოზი. "
        "ან მარტივი წესი უფრო ზუსტი აღმოჩნდა, და მაშინ ისინი საერთოდ არ უნდა იქნას "
        "გამოყენებული, ან მოდელმა ვერ აჩვენა, რომ ცალკეულ დღეებს წინასწარ განჭვრეტს, და "
        "მაშინ ისინი მხოლოდ ტიპიური დონის ორიენტირია.",

    "The low end of the published range. The actual figure should fall below it about one "
    "day in ten.":
        "გამოქვეყნებული დიაპაზონის ქვედა ზღვარი. ფაქტობრივი ციფრი მასზე დაბლა უნდა "
        "მოხვდეს დაახლოებით ათიდან ერთ დღეს.",

    "The central estimate. The actual figure should fall above it about as often as "
    "below it.":
        "ცენტრალური შეფასება. ფაქტობრივი ციფრი მასზე მაღლა დაახლოებით იმდენჯერ უნდა "
        "მოხვდეს, რამდენჯერაც დაბლა.",

    "The high end of the published range. The actual figure should fall above it about "
    "one day in ten.":
        "გამოქვეყნებული დიაპაზონის ზედა ზღვარი. ფაქტობრივი ციფრი მასზე მაღლა უნდა "
        "მოხვდეს დაახლოებით ათიდან ერთ დღეს.",

    "A published forecast whose day has not been reported yet, so there is no actual "
    "figure to score it against. It is listed rather than hidden.":
        "გამოქვეყნებული პროგნოზი, რომლის დღეც ჯერ არ არის დაფიქსირებული, ამიტომ არ "
        "არსებობს ფაქტობრივი ციფრი მის შესაფასებლად. იგი ჩამოთვლილია და არა დამალული.",

    # ── The verdicts ────────────────────────────────────────────────────────
    "usable as a forecast": "გამოსადეგია როგორც პროგნოზი",
    "shown as a guide only": "ნაჩვენებია მხოლოდ როგორც ორიენტირი",
    "not usable": "გამოუსადეგარია",
    "measured": "გაზომილი",
    "UNTESTED": "შეუმოწმებელი",
    "reference rule": "საორიენტაციო წესი",
    "not installed here": "აქ არ არის დაინსტალირებული",

    # ── The two doors, on the guide page ────────────────────────────────────
    "The two doors": "ორი კარი",
    "Everything here is either official or exploratory, and they never mix":
        "აქ ყველაფერი ან ოფიციალურია, ან საძიებო, და ისინი არასოდეს ერევა ერთმანეთში",
    "What it is.": "რა არის ეს.",
    "How the model was chosen.": "როგორ შეირჩა მოდელი.",
    "What it went through.": "რა გაიარა მან.",
    "What you may do with it.": "რისთვის შეგიძლიათ გამოიყენოთ.",
    "Where.": "სად.",
    "What you can do there.": "რა შეგიძლიათ იქ გააკეთოთ.",
    "One thing to try.": "ერთი რამ, რაც უნდა სცადოთ.",
    "What each page is for, what you can do there, and one thing to try":
        "რისთვის არის თითოეული გვერდი, რა შეგიძლიათ იქ გააკეთოთ და ერთი რამ, რაც უნდა "
        "სცადოთ",

    "Every number this Lab produces comes through one of two doors. Which door it came through "
    "determines what you may do with it.":
        "ყოველი ციფრი, რომელსაც ეს ლაბორატორია აწარმოებს, ორი კარიდან ერთ-ერთით მოდის. "
        "რომელი კარიდან მოვიდა, განსაზღვრავს, თუ რისთვის შეგიძლიათ მისი გამოყენება.",

    "This Lab forecasts daily Treasury cash lines, and it is built so that every number "
    "it shows can be checked. Nothing is published unless it has been measured on days "
    "the model was never shown, and where a model failed a check the Lab says so rather "
    "than quietly leaving it out.":
        "ეს ლაბორატორია პროგნოზირებს ხაზინის დღიურ ფულად ხაზებს და აგებულია ისე, რომ "
        "ყოველი ნაჩვენები ციფრი შემოწმებადი იყოს. არაფერი ქვეყნდება, თუ ის არ არის "
        "გაზომილი იმ დღეებზე, რომლებიც მოდელს არასოდეს უნახავს. თუ მოდელმა შემოწმება "
        "ვერ გაიარა, ლაბორატორია ამას აცხადებს და არ ტოვებს ჩუმად.",

    # ── The standing honesty note ───────────────────────────────────────────
    "Everything in this Lab is honestly evaluated on data the models were held back "
    "from. Nothing here has been proven in production, and no page claims otherwise.":
        "ამ ლაბორატორიაში ყველაფერი პატიოსნად არის შეფასებული იმ მონაცემებზე, რომლებიც "
        "მოდელებს დაფარული ჰქონდათ. აქ არაფერია დადასტურებული რეალურ ექსპლუატაციაში და "
        "არცერთი გვერდი არ ამტკიცებს საწინააღმდეგოს.",


    # ── Verdict banners on the Forecast page ────────────────────────────────
    "**Usable as a forecast.** Every check this model was put through passed, so "
    "the numbers below are the best estimate we have for these days.":
        "**გამოსადეგია როგორც პროგნოზი.** ამ მოდელმა ყველა შემოწმება გაიარა, ამიტომ ქვემოთ "
        "მოცემული ციფრები საუკეთესო შეფასებაა, რაც ამ დღეებისთვის გვაქვს.",

    "**Do not use these numbers.** A simple rule of thumb was more accurate than "
    "this model on data it had never seen, so acting on its figures would be "
    "worse than acting on the rule of thumb.":
        "**ნუ გამოიყენებთ ამ ციფრებს.** მარტივი წესი უფრო ზუსტი აღმოჩნდა, ვიდრე ეს მოდელი "
        "იმ მონაცემებზე, რომლებიც მას არასოდეს უნახავს. ამიტომ მისი ციფრებით მოქმედება "
        "უარესი იქნება, ვიდრე მარტივი წესით მოქმედება.",

    "**Shown as a guide to the typical level, not as a forecast.** The model is "
    "about as accurate as we would want, but it could not show that it "
    "anticipates individual days rather than tracking the usual level, so we "
    "will not call it a forecast.":
        "**ნაჩვენებია როგორც ტიპიური დონის ორიენტირი და არა როგორც პროგნოზი.** მოდელი "
        "დაახლოებით იმდენად ზუსტია, რამდენადაც გვსურდა, მაგრამ მან ვერ აჩვენა, რომ ცალკეულ "
        "დღეებს წინასწარ განჭვრეტს და არა უბრალოდ ჩვეულ დონეს მიჰყვება. ამიტომ ჩვენ მას "
        "პროგნოზს არ ვუწოდებთ.",

    # ── The shared tooltips ─────────────────────────────────────────────────

    "A self-test for whether the inputs actually inform the target. We shuffle the "
    "historical answers, refit, and see how much worse the model gets. If the inputs "
    "carry real information, destroying the link should hurt badly. We require the error "
    "to get at least 1.15 times worse; below that we treat the model as tracking a "
    "typical level rather than anticipating individual days. 1.15 is not a round number "
    "chosen for comfort: it was measured against 360 runs on deliberately scrambled "
    "inputs, whose highest ratio was 1.12, so nothing that passes it can be explained "
    "by noise.":
        "თვითშემოწმება იმისა, ნამდვილად იძლევა თუ არა შემავალი მონაცემები ინფორმაციას "
        "სამიზნეზე. ჩვენ ვურევთ ისტორიულ პასუხებს, თავიდან ვარგებთ მოდელს და ვხედავთ, "
        "რამდენად უარესდება იგი. თუ შემავალი მონაცემები ნამდვილ ინფორმაციას შეიცავს, ამ "
        "კავშირის განადგურებამ მკვეთრად უნდა დააზიანოს შედეგი. ჩვენ მოვითხოვთ, რომ შეცდომა "
        "სულ მცირე 1.15-ჯერ გაუარესდეს. ამაზე ნაკლების შემთხვევაში მოდელს განვიხილავთ "
        "როგორც ტიპიური დონის მიმდევარს და არა ცალკეული დღეების განმჭვრეტს. 1.15 არ არის "
        "მოხერხებულობისთვის არჩეული მრგვალი რიცხვი: იგი გაიზომა 360 გაშვებაზე განზრახ "
        "არეულ მონაცემებზე, სადაც უმაღლესი თანაფარდობა 1.12 იყო, ამიტომ ვერაფერი, რაც ამას "
        "გადალახავს, ვერ აიხსნება ხმაურით.",

    "Error divided by the error of a simple seasonal repeat, measured on the training "
    "period. Below 1.00 means better than that simple rule; above 1.00 means worse. "
    "Unlike a percentage error it stays meaningful on days when the actual value is near "
    "zero, which is why it replaces MAPE on the daily flow targets.":
        "შეცდომა გაყოფილი მარტივი სეზონური გამეორების შეცდომაზე, გაზომილი სასწავლო "
        "პერიოდზე. 1.00-ზე ნაკლები ნიშნავს, რომ ის ჯობია ამ მარტივ წესს, ხოლო 1.00-ზე მეტი "
        "ნიშნავს, რომ ჩამორჩება. პროცენტული შეცდომისგან განსხვავებით იგი აზრიანი რჩება იმ "
        "დღეებშიც, როცა ფაქტობრივი მნიშვნელობა ნულთან ახლოსაა, ამიტომაც ცვლის იგი MAPE-ს "
        "დღიურ ნაკადურ სამიზნეებზე.",

    "The share of actual values that landed inside the predicted range. If a range is "
    "advertised as covering 8 days in 10, coverage should be close to 80%. Well below "
    "that means the range is too narrow and understates risk.":
        "ფაქტობრივი მნიშვნელობების წილი, რომლებიც ნაპროგნოზები დიაპაზონის შიგნით მოხვდა. "
        "თუ დიაპაზონი გამოცხადებულია როგორც ათიდან რვა დღის მომცველი, დაფარვა 80%-თან ახლოს "
        "უნდა იყოს. ამაზე მნიშვნელოვნად ნაკლები ნიშნავს, რომ დიაპაზონი ძალიან ვიწროა და "
        "რისკს აკნინებს.",


    "The single shared benchmark: predict that the value five working days ago repeats. "
    "One implementation is used by every model family so that skill numbers are "
    "comparable. It is deliberately simple, and beating it is a floor rather than an "
    "achievement.":
        "ერთადერთი საერთო საზომი: იწინასწარმეტყველე, რომ ხუთი სამუშაო დღის წინანდელი "
        "მნიშვნელობა მეორდება. ერთი და იგივე რეალიზაცია გამოიყენება ყველა მოდელის ოჯახის "
        "მიერ, რათა უპირატესობის რიცხვები შედარებადი იყოს. იგი განზრახ მარტივია და მისი "
        "დამარცხება მინიმალური ზღვარია და არა მიღწევა.",

    "Validation error divided by training error. A model far better on data it trained "
    "on than on data it did not has memorised rather than learned. Above the gate "
    "threshold the model is excluded from best-model selection.":
        "სავალიდაციო შეცდომა გაყოფილი სასწავლო შეცდომაზე. მოდელი, რომელიც ბევრად უკეთესია "
        "იმ მონაცემებზე, რომლებზეც ივარჯიშა, ვიდრე იმაზე, რომლებზეც არა, დაიმახსოვრა და არ "
        "ისწავლა. ზღვარს ზემოთ მოდელი გამოირიცხება საუკეთესო მოდელის შერჩევიდან.",

    "What do these words mean?": "რას ნიშნავს ეს სიტყვები?",
    "What does this mean?": "რას ნიშნავს ეს?",
}
