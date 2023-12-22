const bobMonolog = [
    "Hello there, fellow explorers of this vast digital universe! I'm Bob, your guide and companion in the enthralling world of Wikispeedia.",
    "Imagine, if you will, a place where information stretches as far and wide as stars in a galaxy. That place, my friends, is Wikipedia. And within this boundless expanse of knowledge, lies our adventure.",
    "Wikispeedia isn't just any ordinary game. Oh no, it's a microcosm of our collective curiosity and intellect.",
    "In this game, we navigate from one Wikipedia article to another, seemingly unrelated one, using only the hyperlinks as our pathways.",
    "It's like a digital maze, a condensed version of Wikipedia with over four thousand articles, each a unique playground for the mind.",
    "You see, my adventures in Wikispeedia are more than just a way to pass time. They're a quest – a quest to uncover the hidden pathways of knowledge, to delve deep into the interconnected web of information.",
    "Each journey I undertake is a discovery, a revelation of the vast diversity that Wikipedia holds.",
    "But what really lies at the heart of these adventures? Is it the skill in finding the quickest path, or the strategy born from intuition and understanding of the semantic links between articles?",
    "This, my friends, is what we seek to uncover in our data story.",
    "As we analyze the gameplay patterns and strategies, not just mine but of many other Wikispeedia enthusiasts, we're uncovering some truly fascinating insights.",
    "We're going beyond just the surface, exploring how our navigation choices are intertwined with the rich semantic content of Wikipedia.",
    "We're dissecting paths, extracting features, and revealing the myriad strategies at play.",
    "Our journey is more than a narrative of digital escapades. It's a deep dive into the human mind's capability to navigate and connect within a sea of information.",
    "Together, we'll journey through the virtual corridors of Wikipedia, gaining a deeper appreciation of the intricate and often surprising ways in which knowledge is linked.",
    "So, join me on this journey. Let's unravel the mysteries of Wikispeedia together.",
    "Each click, each article, each link we explore is a step towards a greater understanding of how we, as curious beings, interact with and make sense of the vast landscape of information that lies at our fingertips.",
    "Are you ready? Let the adventure begin!"
];


document.addEventListener("DOMContentLoaded", function() {
const dialogBox = document.getElementById("scrollDialog");
const dialogText = document.getElementById("scrollDialogText");
const dialogBg = document.getElementById("dialogBackground");
const content = document.getElementById("content");
const section0 = document.getElementById("home");
const sections = document.querySelectorAll('.content');
let dialog = false;
let section_index = 0;

function scrollToSection(section) {
    section = document.getElementById(section);
    const sectionTop = section.offsetTop;
    window.scrollTo(0, sectionTop);
}  

function dialogActivate() {
    // Hide content
    content.classList.add("hidden");
    dialogBox.classList.remove("hidden");
    dialogBg.classList.remove("hidden");
    dialogBg.style.height = bobMonolog.length * 300 + "px";
    
}

function dialogDeactivate() {
    // 
    content.classList.remove("hidden");
    dialogBox.classList.add("hidden");
    dialogBg.classList.add("hidden");
}

function dialogScroll(index) {
    // use the scroll to show the list of dialog inside the dialogText
    //if the index is less than the length of the array, then add the next dialog
    if (index < bobMonolog.length) {
        dialogText.textContent = bobMonolog[index];
    } else {
        //if the index is greater than the length of the array, then remove the dialog box
        section_index += 1;
        dialog = false;
        dialogDeactivate();
        scrollToSection(sections[section_index].id);
        
    }
}

window.addEventListener("scroll", function() {
    //if the window reach the section1, then activate the dialog box
    offSet = section0.offsetHeight+section0.offsetTop
    if(window.scrollY >= offSet && window.scrollY <= offSet+50){
        dialog = true;
        section_index = 0;
    }
    else if(!dialog){
        dialogDeactivate();
    }
    // scroll through the dialog
    if (dialog) {
        //console.log(window.scrollY-dialogBg.offsetTop);
        let index = Math.floor((window.scrollY-dialogBg.offsetTop)/200);
        //console.log(index);
        if (index >= 0){
            console.log(index);
            dialogActivate();
            dialogScroll(index);
        }
    }
    
}


);



});







