const bobMonolog = [
    "Hello there, fellow explorers of this vast digital universe! I'm Bob, your guide and companion in the enthralling world of Wikispeedia.",
    "Imagine, if you will, a place where information stretches as far and wide as stars in a galaxy. That place, my friends, is Wikipedia. And within this boundless expanse of knowledge, lies our adventure.",
    "Wikispeedia isn't just any ordinary game. Oh no, it's a window into humanity's collective curiosity and knowledge.",
    "In this game, we navigate from one Wikipedia article to a target article, seemingly unrelated, using only the hyperlinks as our pathways.",
    "It's like a digital maze, a condensed version of Wikipedia with over four thousand articles, each a unique playground for the mind.",
    "You see, my adventures in Wikispeedia are more than just a way to pass time. They're a quest – a quest to uncover the hidden pathways of knowledge, to delve deep into the interconnected web of information.",
    // "Each journey I undertake is a discovery, a revelation of the vast diversity that Wikipedia holds.",
    "But what really lies at the heart of these adventures? Is it the skill in finding the quickest path, or the strategy born from intuition and understanding of the semantic links between articles?",
    "This, my friends, is what we seek to uncover in our data story.",
    // "As we analyze the gameplay patterns and strategies, not just mine but of many other Wikispeedia enthusiasts, we're uncovering some truly fascinating insights.",
    // "We're going beyond just the surface, exploring how our navigation choices are intertwined with the rich semantic content of Wikipedia.",
    // "We're dissecting paths, extracting features, and revealing the myriad strategies at play.",
    // "Our journey is more than a narrative of digital escapades. It's a deep dive into the human mind's capability to navigate and connect within a sea of information.",
    // "Together, we'll journey through the virtual corridors of Wikipedia, gaining a deeper appreciation of the intricate and often surprising ways in which knowledge is linked.",
    // "So, join me on this journey. Let's unravel the mysteries of Wikispeedia together.",
    "Each click, each article, each link we explore is a step towards a greater understanding of how we, as curious beings, interact with and make sense of the vast landscape of information that lies at our fingertips.",
    "Are you ready? Let the adventure begin!"
];

const bobMonolog2 = [
    "And now, as we delve deeper into the mysteries of data and its transformations, I pass the torch to a figure who stands as a beacon in the world of computation and analysis.",
    "Fellow explorers, I introduce to you Ada Lovelace, a pioneer in our field, who will guide us through the next phase of our journey."
];

const adaMonolog = [
    "Thank you, Bob. Hello, curious minds and brave navigators of this vast digital expanse! I am Ada Lovelace, and it is my pleasure to lead you through this intricate ballet of data analysis.",
    "As we examine our dataset, we encounter features in the first column that are strongly heavy-tailed.",
    "To harness them, we shall employ a technique from our analytical armory - the logarithmic transformation.",
    "However, we encounter an obstacle with the 'backtrack' feature, situated in the second row, first column.",
    "This feature, discrete and steadfast in its few values, eludes the smoothing grace of our logarithmic arcanes.",
    "Our next step is normalization, specifically z-score normalization.",
    // "Imagine this as a way to equalize the heights of mountains and depths of valleys in our data landscape.",
    "We shall also address the NaN values, as we do not wish to leave any gaps in our data.",
    "Instead, we fill these gaps with the mean of each feature - making them meaningful yet unremarkable to our analysis.",
    // "Through this process, we do more than just manipulate numbers.",
    // "We engage in a dance with data, leading it gracefully to reveal its hidden patterns and secrets.",
    // "So, join me, as we continue to weave through the tapestry of information, each step a discovery, each transformation a revelation.",
    "We are now ready to uncover the stories hidden within the numbers!",
    // , exploring the vast landscapes of knowledge that await us.",
    // "Let us embark on this enlightening journey, unraveling the mysteries of data with every analysis we perform.",
    // "The adventure beckons!"
    // Yeesha
    // Last night your mother had a dream...
    // We know that some futures are not cast, by writer or Maker, but the dream tells that D'ni will grow again someday. New seekers of D'ni will flow in from the desert feeling called to something they do not understand.
    // But the dream also tells of a desert bird with the power to weave this new D'ni's future. We fear such power - it changes people.
    // Yeesha, our desert bird, your search seems to take you further and further from us. I hope what you find will bring you closer. 
];

const bobMonolog3 = [
    "With Ada's wisdom and our collective curiosity, there's no limit to the insights we can uncover.",
    "Let's venture forth!"
];


dialogs_people = [
    ["B",[["B",bobMonolog]]],
    ["AB",[["B",bobMonolog2],["A",adaMonolog],["B",bobMonolog3]]]
];

breakpoints_section_after = [1,3,4,5];

document.addEventListener("DOMContentLoaded", function() {
const dialogBoxBob = document.getElementById("scrollDialogBob");
const dialogTextBob = document.getElementById("scrollDialogTextBob");
const dialogBoxAda = document.getElementById("scrollDialogAda");
const dialogTextAda = document.getElementById("scrollDialogTextAda");
const dialogBg = document.getElementById("dialogBackground");
const content = document.getElementById("content");
const breakpoint = document.querySelectorAll('.breakpoint');
const sections = document.querySelectorAll('.content');
let dialog = false;
let breakpoint_index = 0;

function scrollToSection(section) {
    section = document.getElementById(section);
    const sectionTop = section.offsetTop;
    window.scrollTo(0, sectionTop-10);
}  

function dialogActivate(index,peoples) {
    
    //content.classList.add("hidden");
    if(peoples == "B" || peoples == "AB"){
        dialogBoxBob.classList.remove("hidden");
    }
    if(peoples == "A" || peoples == "AB"){
        dialogBoxAda.classList.remove("hidden");
    }
    dialogBg.classList.remove("hidden");
    max_length = dialogs_people[index][1][0][1].length;
    dialogs_people[index][1].forEach(element => {
        length = element[1].length;
        if(length > max_length){
            max_length = length;
        }
    });
    
    //dialogBg.style.height = max_length * 500 + "px";
    
}

function writeDialog(text,people) {
    if(people == "B"){
        dialogTextBob.textContent = text;
        dialogTextAda.textContent = "";
    }
    if(people == "A"){
        dialogTextAda.textContent = text;
        dialogTextBob.textContent = "";
    }
}

function dialogDeactivate() {
    //content.classList.remove("hidden");
    dialogBoxBob.classList.add("hidden");
    dialogBoxAda.classList.add("hidden");
    //dialogBg.classList.add("hidden");
}

function dialogScroll(index, dialogs,current_dialog_index) {
    // use the scroll to show the list of dialog inside the dialogTextBob
    //if the index is less than the length of the array, then add the next dialog

    if(current_dialog_index < dialogs.length){
        if (index < dialogs[current_dialog_index][1].length) {
            writeDialog(dialogs[current_dialog_index][1][index],dialogs[current_dialog_index][0]);

            return [current_dialog_index, index];
        } else {

            c = current_dialog_index + 1;
            return [c, 0];
        }
    }
    else{
        dialog = false;
        //if the index is greater than the length of the array, then remove the dialog box
        dialogDeactivate();
        // scrollToSection(sections[breakpoints_section_after[breakpoint_index]].id);
        breakpoint_index = breakpoint_index + 1;
        setTimeout(function(){dialog = false;}, 50);
        return [current_dialog_index, 0];
    }
}

function parallax(element,pos,speed) {
    let offSet = window.scrollY-pos/2;
    element.style.transform = `translateY(${offSet * speed}px)`;
}

window.addEventListener("scroll", function() {
    //if the window reach the section1, then activate the dialog box
    offSet = breakpoint[breakpoint_index].offsetTop + breakpoint[breakpoint_index].offsetHeight;
    if(window.scrollY >= offSet && window.scrollY <= offSet+150 && !dialog){
        console.log("dialog");
        dialog = true;
        index = 0;
        current_dialog_index = 0;
        dialogActivate(breakpoint_index, dialogs_people[breakpoint_index][0]);
        // window.scrollToSection(sections[breakpoints_section_after[breakpoint_index]].id);
        pos = window.scrollY;

    }

    else if(!dialog){
        dialogDeactivate();
    }
    
    // scroll through the dialog
    if (dialog) {
        if(index == 0 && this.scrollY < pos){
            pos = window.scrollY;
            index = 0;
        }
        if (this.scrollY-pos > 200) {
            index = index + 1;
            pos = window.scrollY;
        }
        if (this.scrollY-pos < -200) {
            index = index - 1;
            pos = window.scrollY;
        }
        
        parallax(content,pos,0.7);
        
        
        if (index <= -1) {
            // scrollToSection(sections[(breakpoints_section_after[breakpoint_index])-1].id);
            dialogDeactivate();
            dialog = false;
            index = 0;
        }
        else{
            r = dialogScroll(index, dialogs_people[breakpoint_index][1],current_dialog_index);
            current_dialog_index = r[0];
            index = r[1];

        }
    }
    
}


);



});







