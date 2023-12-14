function scrollToSection(sectionId) {
    document.getElementById(sectionId).scrollIntoView({ behavior: 'smooth' });
  }
  
  window.addEventListener('scroll', function() {
    let sections = document.querySelectorAll('section');
    let dots = document.querySelectorAll('.dot');
  
    sections.forEach((section, index) => {
      let top = window.scrollY;
      let offset = section.offsetTop;
      let height = section.offsetHeight;
  
      if (top >= offset && top < offset + height) {
        dots.forEach((dot) => { dot.classList.remove('active'); });
        dots[index].classList.add('active');
      }
    });
  });