document.addEventListener('DOMContentLoaded', function() {
  const sections = document.querySelectorAll('.content');
  const dots = document.querySelectorAll('.dot');
  const numSections = sections.length;
  document.querySelector('.content_container').style.height = `${numSections * 100}vh`;

  window.scrollToSection = function(sectionId) {
    const section = document.getElementById(sectionId);
    const index = Array.from(sections).indexOf(section);
    window.scrollTo(0, index * window.innerHeight);
  }

  window.addEventListener('scroll', function() {
    const currentSection = Math.floor(window.scrollY / window.innerHeight);

    sections.forEach((section, index) => {
      if (index === currentSection) {
        section.classList.add('active');
      } else {
        section.classList.remove('active');
      }

      if (index === currentSection) {
        dots.forEach((dot) => dot.classList.remove('active'));
        if (dots[index]) {
          dots[index].classList.add('active');
        }
      }
    });
  });
});