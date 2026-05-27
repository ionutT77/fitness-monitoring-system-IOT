# Why Do We Use These Features?

This document outlines the physiological and biological rationale behind every feature collected by the hardware system and generated for the ML model. By understanding *why* these features matter, we can see exactly how the AI learns to distinguish between a "Maximum Effectiveness" workout and a "Low Effectiveness" workout, regardless of who is working out.

---

## 1. Demographic & Biological Context (The Baseline)

Before a single drop of sweat falls, the model needs to know *who* is working out. A heart rate of 160 BPM means something completely different for a 20-year-old athlete versus a 60-year-old beginner.

### Age
*   **What it affects:** Primarily limits **Max HR**.
*   **Why we use it:** Maximum heart rate is dictated by age and genetics, not fitness level (roughly `220 - Age`). Without age context, the AI might think a 55-year-old hitting 160 BPM is "slacking off" compared to a 20-year-old hitting 190 BPM. Age allows the AI to normalize the effort relative to the user's biological ceiling.

### Fitness Level (`low`, `medium`, `high`)
*   **What it affects:** Stroke volume, HR recovery, and Neuromuscular Efficiency.
*   **Why we use it:** As you get fitter, your heart pumps more blood per beat (stroke volume). A highly trained athlete will have a *lower* `avg_hr` and fewer `hr_spikes` for the exact same physical work as a beginner. Simultaneously, an advanced lifter's brain is better at recruiting motor units (Neuromuscular Efficiency), leading to *higher* initial `avg_emg` signals. Without knowing fitness level, the model might penalize a fit person for having a low heart rate!

### Athlete Type (`powerlifter`, `hybrid`, `gym_bro`, `non_athletic`)
*   **What it affects:** Rests, HR Spikes, Rep Volume.
*   **Why we use it:** A powerlifter doing heavy singles will naturally take 5-minute rests. Their `avg_hr` will be low, and their `total_reps` will be minimal, but their `avg_emg` will be extreme. This category provides the AI with behavioral context so it doesn't mistakenly classify powerlifting as "Low Effectiveness" just because the heart rate was low.

### Workout Type (`HILV`, `LIHV`, `hypertrophy`, `endurance_lifting`)
*   **What it affects:** Expected Volume, Target Heart Rate, and EMG Expectations.
*   **Why we use it:** 5 hard reps are better than 20 lazy reps. Volume does not necessarily equal intensity. By telling the AI the *intent* of the resistance training session, it stops blindly correlating `total_reps` with effectiveness. 
    *   **HILV (High Intensity, Low Volume):** e.g., Powerlifting or 1-3 rep maxes. The AI will expect very few reps and long rests, but demands extreme `avg_emg`.
    *   **LIHV (Low Intensity, High Volume):** e.g., Light pump work or rehab. The AI expects high reps, but knows the EMG will be lower.
    *   **Hypertrophy:** Standard bodybuilding (8-12 reps). Expects moderate spikes and high fatigue.
    *   **Endurance Lifting:** e.g., CrossFit-style barbell cycling. Expects massive reps and high heart rate, but lower peak EMG.
*(Note: We restrict this model strictly to resistance training, ignoring pure cardio like running/rowing).*

### Body Fat Percentage
*   **What it affects:** Cardiovascular load (`avg_hr`).
*   **Why we use it:** Moving a higher body mass requires more oxygen. Someone with higher body fat will naturally sit at a higher heart rate when performing identical exercises compared to a leaner person. This prevents the model from assuming "High HR = Elite Intensity" when it might just be the result of moving extra mass.

### Limb Length (`short`, `medium`, `long`)
*   **What it affects:** Mechanical work, `emg_fatigue`, `total_reps`.
*   **Why we use it:** Leverages matter. Someone with long arms has to physically push a bench press bar further. This requires more mechanical work per rep, which induces earlier `emg_fatigue` and lowers `total_reps`. By feeding the model limb length, it learns to excuse a lower rep count from a tall, long-limbed athlete.

---

## 2. Cardiovascular Metrics (The Engine)

These metrics evaluate the aerobic demand and rest periods of the workout.

### Duration (minutes)
*   **Why we use it:** Time under tension. A workout that lasts 3 minutes cannot physically be a "High" effectiveness workout, no matter how hard you push. Conversely, a 90-minute workout of just sitting around isn't effective either. Duration gives the model the time-horizon to evaluate the other metrics against.

### Average Heart Rate (`avg_hr`)
*   **Why we use it:** Measures sustained effort. However, it *must* be analyzed alongside the demographic features. A high `avg_hr` generally indicates good effort, but if it is too high for too long, it might just indicate a lack of conditioning (poor fitness level) rather than an elite workout.

### Maximum Heart Rate (`max_hr`)
*   **Why we use it:** Shows the absolute peak intensity the user reached. While `avg_hr` can be lowered by taking long rests, `max_hr` proves that the user pushed themselves to their limit during at least one set.

### HR Spikes (`hr_spikes`)
*   **Why we use it:** This is a proxy for the number of sets or high-intensity intervals performed. When you rest, your heart rate drops. When you start a set, it spikes. Counting spikes gives the model a silent look at the structure of your workout.

### Percent Time Low (`pct_time_low`)
*   **Why we use it:** Measures Recovery and Aerobic Conditioning. If this percentage is huge, you spent the whole gym session scrolling on your phone. If it is tiny, you were doing relentless circuits. Importantly, how quickly your heart rate drops back into this low zone after a spike is a direct measure of your parasympathetic nervous system function.

---

## 3. Muscular Metrics (The Force)

These metrics evaluate central nervous system output and muscle fiber recruitment.

### Average EMG (`avg_emg`)
*   **Why we use it:** This is the purest measure of effort. Heart rate can lie (due to caffeine, stress, or heat), but electrical muscle activation does not. Higher EMG means more motor units (muscle fibers) were recruited. Thanks to the `fitness_level` context, the model knows that trained weightlifters will naturally produce higher EMG signals because their brains are more efficient at recruiting muscle fibers.

### EMG Fatigue (`emg_fatigue`)
*   **Why we use it:** Measures how much your muscle activation amplitude dropped from the beginning of the workout to the end. Fast-twitch (Type II) muscle fibers are powerful but fatigue quickly. A high `emg_fatigue` proves that you pushed your fast-twitch fibers to their limit and achieved real muscular overload.

### Total Reps (`total_reps`)
*   **Why we use it:** A measure of total volume. However, because it is now combined with `workout_type`, `athlete_type`, and `limb_length`, the model is smart enough to know that a Powerlifter doing 10 reps might be achieving the same "Maximum Effectiveness" as an Endurance athlete doing 200 reps. We specifically suppress the AI's reliance on this feature so that 5 max-effort reps are correctly graded higher than 20 lazy reps.
