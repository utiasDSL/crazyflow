const carousels = [];
let focusedCarousel = null;

function initializeCarousel(carouselContainer) {
  if (carouselContainer.dataset.carouselReady === "true") {
    return;
  }

  carouselContainer.dataset.carouselReady = "true";

  const carousel = carouselContainer.querySelector(".carousel");
  const slides = Array.from(carouselContainer.querySelectorAll(".carousel-slide"));
  const prevBtn = carouselContainer.querySelector(".carousel-btn-prev");
  const nextBtn = carouselContainer.querySelector(".carousel-btn-next");
  const indicatorsContainer = carouselContainer.querySelector(".carousel-indicators");

  if (!carousel || slides.length === 0 || !indicatorsContainer) {
    return;
  }

  let currentSlide = 0;

  slides.forEach((_, index) => {
    const indicator = document.createElement("button");
    indicator.type = "button";
    indicator.classList.add("carousel-indicator");
    indicator.setAttribute("aria-label", `Go to slide ${index + 1}`);
    indicator.addEventListener("click", () => goToSlide(index));
    indicatorsContainer.appendChild(indicator);
  });

  const indicators = Array.from(indicatorsContainer.querySelectorAll(".carousel-indicator"));

  function goToSlide(index) {
    slides.forEach((slide) => {
      slide.classList.remove("active", "prev", "next");
    });

    currentSlide = (index + slides.length) % slides.length;

    const prevIndex = (currentSlide - 1 + slides.length) % slides.length;
    const nextIndex = (currentSlide + 1) % slides.length;

    slides[currentSlide].classList.add("active");
    slides[prevIndex].classList.add("prev");
    slides[nextIndex].classList.add("next");

    indicators.forEach((indicator, indicatorIndex) => {
      indicator.classList.toggle("active", indicatorIndex === currentSlide);
      indicator.setAttribute("aria-current", indicatorIndex === currentSlide ? "true" : "false");
    });
  }

  function nextSlide() {
    goToSlide(currentSlide + 1);
  }

  function prevSlide() {
    goToSlide(currentSlide - 1);
  }

  if (prevBtn) {
    prevBtn.addEventListener("click", prevSlide);
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", nextSlide);
  }

  const api = { container: carouselContainer, nextSlide, prevSlide };
  const focus = () => {
    focusedCarousel = api;
  };
  carouselContainer.addEventListener("pointerenter", focus);
  carouselContainer.addEventListener("focusin", focus);

  let touchStartX = 0;
  let touchEndX = 0;

  carousel.addEventListener(
    "touchstart",
    (event) => {
      focus();
      touchStartX = event.changedTouches[0].screenX;
    },
    { passive: true }
  );

  carousel.addEventListener(
    "touchend",
    (event) => {
      touchEndX = event.changedTouches[0].screenX;
      handleSwipe();
    },
    { passive: true }
  );

  function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartX - touchEndX;

    if (Math.abs(diff) <= swipeThreshold) {
      return;
    }

    if (diff > 0) {
      nextSlide();
    } else {
      prevSlide();
    }
  }

  carousels.push(api);
  goToSlide(0);
}

function initializeCarousels() {
  for (let i = carousels.length - 1; i >= 0; i--) {
    if (!carousels[i].container.isConnected) {
      carousels.splice(i, 1);
    }
  }
  document.querySelectorAll(".carousel-container").forEach(initializeCarousel);
}

document.addEventListener("keydown", (event) => {
  if (carousels.length === 0 || (event.key !== "ArrowLeft" && event.key !== "ArrowRight")) {
    return;
  }

  const target = carousels.includes(focusedCarousel) ? focusedCarousel : carousels[0];

  if (event.key === "ArrowLeft") {
    target.prevSlide();
  } else {
    target.nextSlide();
  }
});

if (typeof document$ !== "undefined") {
  document$.subscribe(initializeCarousels);
} else if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeCarousels);
} else {
  initializeCarousels();
}
