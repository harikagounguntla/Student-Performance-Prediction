document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const submitBtn = document.querySelector('button[type="submit"]');

    // Safety Check: Only run if the form actually exists on this page
    if (form) {
        form.addEventListener('submit', function(e) {
            
            // 1. Get elements safely
            const mathInput = document.getElementsByName('math_score')[0];
            const readingInput = document.getElementsByName('reading_score')[0];

            // 2. Convert to numbers (default to 0 if empty)
            const mathScore = mathInput ? parseFloat(mathInput.value) : 0;
            const readingScore = readingInput ? parseFloat(readingInput.value) : 0;

            // 3. Validation Logic (Handles NaN and Range)
            if (isNaN(mathScore) || isNaN(readingScore) || 
                mathScore < 0 || mathScore > 100 || 
                readingScore < 0 || readingScore > 100) {
                
                alert("⚠️ Error: Please enter valid Math and Reading scores between 0 and 100.");
                e.preventDefault(); 
                return;
            }

            // 4. Visual Feedback
            if (submitBtn) {
                submitBtn.innerHTML = "Predicting Score... ⏳";
                submitBtn.style.backgroundColor = "#636e72";
                submitBtn.style.cursor = "not-allowed";
                submitBtn.disabled = true; 
            }
        });
    }
});