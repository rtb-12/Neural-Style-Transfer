document
  .getElementById("contentImageInput")
  .addEventListener("change", function (e) {
    var reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById("contentImageDisplay").src = event.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
  });

document
  .getElementById("styleImageInput")
  .addEventListener("change", function (e) {
    var reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById("styleImageDisplay").src = event.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
  });

document
  .getElementById("contentImageInput2")
  .addEventListener("change", function (e) {
    var reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById("contentImageDisplay2").src = event.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
  });

document
  .getElementById("styleImageInput1")
  .addEventListener("change", function (e) {
    var reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById("styleImageDisplay1").src = event.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
  });

document
  .getElementById("styleImageInput2")
  .addEventListener("change", function (e) {
    var reader = new FileReader();
    reader.onload = function (event) {
      document.getElementById("styleImageDisplay2").src = event.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
  });

//first page

function generateImage(event) {
  event.preventDefault(); // Prevent form submission
  var promptInput = document.getElementById("prompt_input").value;

  // Validate the prompt to ensure it contains no numbers
  var promptRegex = /^[^\d]+$/;
  if (!promptRegex.test(promptInput)) {
    alert("Please enter a prompt without numbers.");
    return;
  }

  // Create the request object
  fetch("/generate_image", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    },
    body: "prompt=" + encodeURIComponent(promptInput),
  })
    .then((response) => {
      if (response.status === 200) {
        // Fetch the generated image directly
        return fetch("/generated_style_image");
      } else {
        throw new Error("An error occurred while generating the image.");
      }
    })
    .then((response) => {
      if (response && response.ok) {
        document.getElementById("styleImageDisplay").src =
          "/generated_style_image";
      }
    })
    .catch((error) => {
      alert(error.message);
    });
}

function generateImage1(event) {
  console.log("Generating image 1 ...clicked");
  event.preventDefault(); // Prevent form submission
  var promptInput = document.getElementById("prompt_input1").value;

  // Validate the prompt to ensure it contains no numbers
  var promptRegex = /^[^\d]+$/;
  if (!promptRegex.test(promptInput)) {
    alert("Please enter a prompt without numbers.");
    return;
  }

  // Create the request object
  fetch("/generate_image", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    },
    body: "prompt=" + encodeURIComponent(promptInput),
  })
    .then((response) => {
      if (response.status === 200) {
        // Fetch the generated image directly
        return fetch("/generated_style_image");
      } else {
        throw new Error("An error occurred while generating the image.");
      }
    })
    .then((response) => {
      if (response && response.ok) {
        document.getElementById("styleImageDisplay1").src =
          "/generated_style_image";
      }
    })
    .catch((error) => {
      alert(error.message);
    });
}

function generateImage2(event) {
  console.log("Generating image 2...clicked");
  event.preventDefault(); // Prevent form submission
  var promptInput = document.getElementById("prompt_input2").value;

  // Validate the prompt to ensure it contains no numbers
  var promptRegex = /^[^\d]+$/;
  if (!promptRegex.test(promptInput)) {
    alert("Please enter a prompt without numbers.");
    return;
  }

  // Create the request object
  fetch("/generate_image", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    },
    body: "prompt=" + encodeURIComponent(promptInput),
  })
    .then((response) => {
      if (response.status === 200) {
        // Fetch the generated image directly
        return fetch("/generated_style_image");
      } else {
        throw new Error("An error occurred while generating the image.");
      }
    })
    .then((response) => {
      if (response && response.ok) {
        document.getElementById("styleImageDisplay2").src =
          "/generated_style_image";
      }
    })
    .catch((error) => {
      alert(error.message);
    });
}

function startStyleTransfer(event) {
  console.log("StyleTransfer Started");
  event.preventDefault(); // Prevent form submission
  // Show loading before making the fetch request
  // Show loading spinner

  fetch("/", {
    method: "POST",
    body: new FormData(document.getElementById("styleTransferForm1")),
  })
    .then((response) => {
      if (response.redirected) {
        return fetch("/generated_image");
      }
    })
    .then((response) => {
      if (response && response.ok) {
        document.getElementById("generatedImageDisplay").src =
          "/generated_image";
        // Hide loading after the image is loaded
        // Hide loading spinner
      }
    });
}

function startBackgroundStyleTransfer(event) {
  console.log("BStyleTransfer Started");
  event.preventDefault(); // Prevent form submission
  // Show loading spinner

  fetch("/background_style_transfer", {
    method: "POST",
    body: new FormData(document.getElementById("styleTransferForm1")),
  })
    .then((response) => {
      if (response.redirected) {
        return fetch("/generated_background_image");
      }
    })
    .then((response) => {
      if (response && response.ok) {
        document.getElementById("generatedImageDisplay").src =
          "/generated_background_image";
      }
    });
}

function startStyleTransferColorPreserve(event) {
  console.log("CPtyleTransfer Started");
  event.preventDefault(); // Prevent form submission
  // Show loading spinner

  fetch("/style_transfer_color_preserve", {
    method: "POST",
    body: new FormData(document.getElementById("styleTransferForm1")),
  })
    .then((response) => {
      if (response.redirected) {
        return fetch("/generated_image_color_preserve");
      }
    })
    .then((response) => {
      if (response && response.ok) {
        document.getElementById("generatedImageDisplay").src =
          "/generated_image_color_preserve";
      }
    });
}

function startStyleTransferHighRes(event) {
  console.log("HRtyleTransfer Started");
  event.preventDefault(); // Prevent form submission
  // Show loading spinner

  fetch("/high_resolution_style_transfer", {
    method: "POST",
    body: new FormData(document.getElementById("styleTransferForm1")),
  })
    .then((response) => {
      if (response.redirected) {
        return fetch("/generated_image_high_res");
      }
    })
    .then((response) => {
      if (response && response.ok) {
        document.getElementById("generatedImageDisplay").src =
          "/generated_image_high_res";
      }
    });
}

//second

function startStyleTransfer2(event) {
  console.log("StyleTransfer Started 2");
  event.preventDefault(); // Prevent form submission
  // Show loading before making the fetch request
  // Show loading spinner

  fetch("/second_page", {
    method: "POST",
    body: new FormData(document.getElementById("styleTransferForm2")),
  })
    .then((response) => {
      if (response.redirected) {
        return fetch("/generated_image2");
      }
    })
    .then((response) => {
      if (response && response.ok) {
        document.getElementById("generatedImageDisplay2").src =
          "/generated_image2";
        // Hide loading after the image is loaded
        // Hide loading spinner
      }
    });
}

function startBackgroundStyleTransfer2(event) {
  console.log("BStyleTransfer Started2");
  event.preventDefault(); // Prevent form submission
  // Show loading spinner

  fetch("/background_style_transfer2", {
    method: "POST",
    body: new FormData(document.getElementById("styleTransferForm2")),
  })
    .then((response) => {
      if (response.redirected) {
        return fetch("/generated_background_image2");
      }
    })
    .then((response) => {
      if (response && response.ok) {
        document.getElementById("generatedImageDisplay2").src =
          "/generated_background_image2";
      }
    });
}

function startStyleTransferColorPreserve2(event) {
  console.log("CPtyleTransfer Started2");
  event.preventDefault(); // Prevent form submission
  // Show loading spinner

  fetch("/style_transfer_color_preserve2", {
    method: "POST",
    body: new FormData(document.getElementById("styleTransferForm2")),
  })
    .then((response) => {
      if (response.redirected) {
        return fetch("/generated_image_color_preserve2");
      }
    })
    .then((response) => {
      if (response && response.ok) {
        document.getElementById("generatedImageDisplay2").src =
          "/generated_image_color_preserve2";
      }
    });
}

function startStyleTransferHighRes2(event) {
  console.log("HRtyleTransfer Started");
  event.preventDefault(); // Prevent form submission
  // Show loading spinner

  fetch("/high_resolution_style_transfer2", {
    method: "POST",
    body: new FormData(document.getElementById("styleTransferForm2")),
  })
    .then((response) => {
      if (response.redirected) {
        return fetch("/generated_image_high_res2");
      }
    })
    .then((response) => {
      if (response && response.ok) {
        document.getElementById("generatedImageDisplay2").src =
          "/generated_image_high_res";
      }
    });
}

function showLoading() {
  var loadingDiv = document.querySelector(".loader");
  loadingDiv.style.display = "block";
  // document.getElementById("generatedImageDisplay").style.display = "none";
}

function hideLoading() {
  var loadingDiv = document.querySelector(".loader");
  loadingDiv.style.display = "none";
  // document.getElementById("generatedImageDisplay").style.display ="block";
}
