
document.addEventListener('DOMContentLoaded', function() {
  const sections = document.querySelectorAll('.content');
  const dots = document.querySelectorAll('.dot');
  const numSections = sections.length;
  const footer = document.querySelector('footer');
  const dotSlider = document.getElementById('dotSlider');

 

  function updateOpacityAndDots() {
    const middleOfScreen = window.innerHeight / 2 + window.scrollY;
    let closestSectionIndex = 0;
    let minDistanceToMiddle = Number.MAX_VALUE;

    sections.forEach((section, index) => {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.offsetHeight;
      const sectionMid = sectionTop + sectionHeight / 2;
      const distanceToMiddle = Math.abs(middleOfScreen - sectionMid);

      // Check for closest section
      if (distanceToMiddle < minDistanceToMiddle) {
        minDistanceToMiddle = distanceToMiddle;
        closestSectionIndex = index;
      }
    });

    if (closestSectionIndex === 0) {
      footer.style.opacity = 0;
      dotSlider.style.opacity = 0;

    }
    else {
      footer.style.opacity = 1;
      dotSlider.style.opacity = 1;
    }

    // Update dots
    dots.forEach((dot, index) => {
      if (index === closestSectionIndex) {
        dot.classList.add('active');
      } else {
        dot.classList.remove('active');
      }
    });
  }

  window.addEventListener('scroll', updateOpacityAndDots);
  updateOpacityAndDots(); // Initial update on page load
});

function scrollToSection(section) {
  section = document.getElementById(section);
  const sectionTop = section.offsetTop;
  window.scrollTo(0, sectionTop-10);

}

function scrollStart(){
  home = document.getElementById("home");
  scrollTo(0,home.offsetHeight+10);
}


