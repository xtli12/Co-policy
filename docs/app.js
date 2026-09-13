/* Interactive figures for the Co-policy project page.
 *
 *   1. Guided self-attention explorer  - paper Figure 2
 *   2. Gaussian-mixture action generator - paper Figure 5(a)
 *   3. Semantic anchor bank            - paper Figure 4
 *
 * All three widgets progressively enhance static markup, so the page stays
 * readable with scripting disabled.
 */
(function () {
  "use strict";

  /* ==================================================================
   * 1. Guided self-attention explorer (Figure 2)
   * ================================================================== */

  /* Each stage names the panels it lights up. The value is the role badge drawn
   * on that panel, or an empty string when the panel needs no badge. */
  var GSA_STAGES = {
    obs: {
      panels: { obs: "" },
      copy:
        "A single egocentric RGB frame is the only visual input, and both encoder " +
        "branches read that same frame. Nothing is discarded by a pre-selected crop " +
        "or a hand-designed region of interest."
    },
    f1: {
      panels: { f1: "" },
      copy:
        "The DenseNet branch produces F\u2081. Its response is patchy and pixel-wise: " +
        "bright regions mark contact-level detail such as bell edges and rim texture, " +
        "which is the evidence a strike has to land on. On its own it fires on many " +
        "edges at once and commits to none of them."
    },
    f2: {
      panels: { f2: "" },
      copy:
        "The Swin Transformer branch produces F\u2082. Shifted-window attention makes it " +
        "smooth and long-range, so it encodes where the chime rack sits and how the rows " +
        "are laid out. It cannot resolve which individual bell is the target."
    },
    align: {
      panels: { f1: "aligned", f2: "aligned" },
      copy:
        "AWCA reweights channel responses so the two branches become comparable, then " +
        "PSNL restores long-range compatibility inside the local stream. Without this " +
        "alignment the two maps are not on a common footing, and averaging one into the " +
        "other destroys both."
    },
    gsa: {
      panels: { f2: "queries Q", f1: "keys + values K, V", f3: "result" },
      copy:
        "Guided self-attention is deliberately asymmetric: queries come from the global " +
        "branch while keys and values come from the local branch. Global context poses the " +
        "question of which bell is the target, and local evidence answers it. In vanilla " +
        "self-attention all three come from one sequence, so the long-range terms dominate."
    },
    f3: {
      panels: { f3: "" },
      copy:
        "The coupled feature F\u2083 commits to one bell. The contrast is the point: the local " +
        "map fires on many edges, the global map covers the whole rack, and only the guided " +
        "map produces a single localized target for the action head to aim at."
    }
  };

  function initGsaExplorer() {
    var root = document.querySelector("[data-gsa-explorer]");
    if (!root) {
      return;
    }

    var chips = toArray(root.querySelectorAll("[data-stage]"));
    var panels = toArray(root.querySelectorAll("[data-panel]"));
    var copy = root.querySelector("[data-stage-copy]");

    function select(stage, moveFocus) {
      var spec = GSA_STAGES[stage];
      if (!spec) {
        return;
      }

      chips.forEach(function (chip) {
        var on = chip.getAttribute("data-stage") === stage;
        chip.setAttribute("aria-checked", on ? "true" : "false");
        chip.tabIndex = on ? 0 : -1;
        if (on && moveFocus) {
          chip.focus();
        }
      });

      panels.forEach(function (panel) {
        var key = panel.getAttribute("data-panel");
        var role = spec.panels[key];
        var on = typeof role === "string";
        panel.classList.toggle("is-active", on);

        var badge = panel.querySelector("[data-badge]");
        if (badge) {
          badge.textContent = on ? role : "";
          badge.hidden = !on || role === "";
        }
      });

      copy.textContent = spec.copy;
    }

    chips.forEach(function (chip, index) {
      chip.addEventListener("click", function () {
        select(chip.getAttribute("data-stage"), false);
      });
      chip.addEventListener("keydown", function (event) {
        var next = radioKeyTarget(event.key, index, chips.length);
        if (next === null) {
          return;
        }
        event.preventDefault();
        select(chips[next].getAttribute("data-stage"), true);
      });
    });

    /* Clicking a feature map is the same action as clicking its chip. */
    panels.forEach(function (panel) {
      var stage = panel.getAttribute("data-panel");
      if (GSA_STAGES[stage]) {
        panel.addEventListener("click", function () {
          select(stage, false);
        });
      }
    });

    root.classList.add("is-interactive");
    select("obs", false);
  }

  /* ==================================================================
   * 2. Gaussian-mixture action generator (Figure 5a)
   * ================================================================== */

  /* Mixture components. The three action modes are the ones named in the paper's
   * Figure 5(a) caption; the weights and positions are schematic, exactly as in
   * the published diagram, and are not measured per-component values. */
  var COMPONENTS = [
    { id: "m1", name: "Top-down strike", color: "#8f6fc0", weight: 0.53, x: 232, y: 250, rx: 50, ry: 40 },
    { id: "m2", name: "Side swing", color: "#d9534f", weight: 0.28, x: 104, y: 130, rx: 42, ry: 34 },
    { id: "m3", name: "Gentle tap", color: "#4c7e4d", weight: 0.19, x: 302, y: 118, rx: 40, ry: 32 }
  ];

  var CANDIDATE = { x: 158, y: 178 }; // GSA-guided candidate, before mode selection

  var RULES = {
    argmax: {
      formula: "\u03c4* = \u03bc_k*,  k* = arg max_k \u03c0_k",
      copy:
        "Taking the most likely component keeps one committed motion. The executable " +
        "action sits inside a mode, so it stays a motion the arm can actually perform."
    },
    expectation: {
      formula: "\u03c4* = \u03a3_k \u03c0_k \u03bc_k",
      copy:
        "Averaging the components lands between the modes, in a region no demonstration " +
        "ever visited. This is the failure deterministic behaviour cloning cannot avoid, " +
        "and it is why the modes are kept separate rather than regressed into one action."
    }
  };

  var STEPS = [
    {
      title: "Encode the observation",
      formula: "h = GSA(O)",
      copy:
        "Guided self-attention turns the RGB frame into one feature vector. The action " +
        "space below is still empty: no motion has been proposed yet."
    },
    {
      title: "One forward pass",
      formula: "{\u03c0_k, \u03bc_k, \u03a3_k} = f_\u03b8(h, N^r, c)",
      copy:
        "A single pass of the mixture-density head emits every component at once: a weight, " +
        "a mean action segment, and a covariance. There is no denoising loop and no EM fit, " +
        "which is what keeps execution at interactive speed."
    },
    {
      title: "Read the candidate modes",
      formula: "\u03bc\u2081, \u03bc\u2082, \u03bc\u2083   (feasible action modes)",
      copy:
        "Each mean is one complete way to reach the same target note over the whole action " +
        "segment. The modes are alternatives, not joints: a top-down strike and a side swing " +
        "sound alike but are different motions."
    },
    {
      title: "Select an executable action",
      formula: null, // supplied by the active selection rule
      copy: null
    },
    {
      title: "Execute",
      formula: "\u03c4* \u2192 robot",
      copy:
        "The chosen segment goes straight to the arm. Perception, mode prediction, and " +
        "selection have each happened once, so the loop runs at 18.6 Hz after a command is " +
        "available."
    }
  ];

  /* The action space and the likelihood chart are separate SVGs so each scales to
   * its own column. A single wide SVG would shrink its labels to a few pixels once
   * the two halves have to share a phone-width column. */
  var SPACE = { x: 8, y: 34, w: 404, h: 300 };
  var BARS = { x: 36, base: 300, top: 118, width: 46, gap: 30 };
  var RING_FACTORS = [1.75, 1.32, 1.0, 0.6];
  var SVG_NS = "http://www.w3.org/2000/svg";

  function initGmmExplorer() {
    var root = document.querySelector("[data-gmm-explorer]");
    if (!root) {
      return;
    }

    var stage = root.querySelector("[data-gmm-stage]");
    var cards = toArray(root.querySelectorAll("[data-component]"));
    var ruleButtons = toArray(root.querySelectorAll("[data-rule]"));
    var stepButtons = toArray(root.querySelectorAll("[data-step]"));
    var playButton = root.querySelector("[data-play]");
    var titleOut = root.querySelector("[data-step-title]");
    var formulaOut = root.querySelector("[data-formula]");
    var copyOut = root.querySelector("[data-step-copy]");
    var readoutOut = root.querySelector("[data-readout]");

    var state = { step: 0, rule: "argmax", active: "m1" };
    var timer = null;
    var scene = buildScene(stage);

    function argmaxId() {
      return COMPONENTS.reduce(function (best, item) {
        return item.weight > best.weight ? item : best;
      }).id;
    }

    function render() {
      var step = state.step;
      var rule = RULES[state.rule];
      var winner = argmaxId();
      var spec = STEPS[step];

      /* --- action space and likelihood bars --- */
      scene.components.forEach(function (node) {
        var isActive = node.id === state.active;
        node.group.classList.toggle("is-visible", step >= 1);
        node.group.classList.toggle("is-active", isActive && step >= 1);
        node.modeStar.classList.toggle("is-visible", step >= 2);

        node.bar.group.classList.toggle("is-visible", step >= 1);
        node.bar.group.classList.toggle("is-active", isActive && step >= 1);
        node.bar.group.classList.toggle(
          "is-chosen",
          step >= 3 && state.rule === "argmax" && node.id === winner
        );
      });

      scene.candidate.classList.toggle("is-visible", step >= 2);

      var chosen = selectedPoint(state.rule, winner);
      scene.selected.setAttribute("transform", "translate(" + chosen.x + "," + chosen.y + ")");
      scene.selected.classList.toggle("is-visible", step >= 3);
      scene.selected.classList.toggle("is-executing", step >= 4);
      scene.gap.classList.toggle("is-visible", step >= 3 && state.rule === "expectation");
      scene.trail.setAttribute(
        "d",
        "M" + CANDIDATE.x + " " + CANDIDATE.y + "L" + chosen.x + " " + chosen.y
      );
      scene.trail.classList.toggle("is-visible", step >= 3);

      /* --- text panels --- */
      titleOut.textContent = String(step + 1) + " / " + STEPS.length + " \u00b7 " + spec.title;
      formulaOut.textContent = spec.formula || rule.formula;
      copyOut.textContent = spec.copy || rule.copy;

      stepButtons.forEach(function (button) {
        var index = Number(button.getAttribute("data-step"));
        button.setAttribute("aria-current", index === step ? "step" : "false");
        button.classList.toggle("is-done", index < step);
      });

      ruleButtons.forEach(function (button) {
        var on = button.getAttribute("data-rule") === state.rule;
        button.setAttribute("aria-checked", on ? "true" : "false");
        button.tabIndex = on ? 0 : -1;
      });
      root.classList.toggle("rule-expectation", state.rule === "expectation");

      /* --- component cards --- */
      var active = componentById(state.active);
      cards.forEach(function (card) {
        var on = card.getAttribute("data-component") === state.active;
        card.setAttribute("aria-checked", on ? "true" : "false");
        card.tabIndex = on ? 0 : -1;
        card.classList.toggle(
          "is-chosen",
          step >= 3 && state.rule === "argmax" && card.getAttribute("data-component") === winner
        );
      });

      readoutOut.textContent =
        step === 0
          ? "no modes predicted yet"
          : active.name +
            "   \u03c0 = " +
            active.weight.toFixed(2) +
            (active.id === winner
              ? "   most likely component"
              : "   alternative kept by the mixture");

      playButton.setAttribute("aria-pressed", timer ? "true" : "false");
      playButton.textContent = timer ? "Pause" : step >= STEPS.length - 1 ? "Replay" : "Play";
    }

    function goTo(step) {
      state.step = Math.max(0, Math.min(STEPS.length - 1, step));
      render();
    }

    function stop() {
      if (timer) {
        window.clearInterval(timer);
        timer = null;
      }
    }

    function play() {
      stop();
      if (state.step >= STEPS.length - 1) {
        state.step = 0;
      }
      timer = window.setInterval(function () {
        if (state.step >= STEPS.length - 1) {
          stop();
          render();
          return;
        }
        goTo(state.step + 1);
      }, 1400);
      render();
    }

    playButton.addEventListener("click", function () {
      if (timer) {
        stop();
        render();
      } else {
        play();
      }
    });

    stepButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        stop();
        goTo(Number(button.getAttribute("data-step")));
      });
    });

    ruleButtons.forEach(function (button, index) {
      button.addEventListener("click", function () {
        state.rule = button.getAttribute("data-rule");
        if (state.step < 3) {
          state.step = 3;
        }
        stop();
        render();
      });
      button.addEventListener("keydown", function (event) {
        var next = radioKeyTarget(event.key, index, ruleButtons.length);
        if (next === null) {
          return;
        }
        event.preventDefault();
        state.rule = ruleButtons[next].getAttribute("data-rule");
        stop();
        render();
        ruleButtons[next].focus();
      });
    });

    function pick(id, moveFocus) {
      state.active = id;
      if (state.step < 1) {
        state.step = 1;
      }
      stop();
      render();
      if (moveFocus) {
        var card = cards.filter(function (item) {
          return item.getAttribute("data-component") === id;
        })[0];
        if (card) {
          card.focus();
        }
      }
    }

    cards.forEach(function (card, index) {
      card.addEventListener("click", function () {
        pick(card.getAttribute("data-component"), false);
      });
      card.addEventListener("keydown", function (event) {
        var next = radioKeyTarget(event.key, index, cards.length);
        if (next === null) {
          return;
        }
        event.preventDefault();
        pick(cards[next].getAttribute("data-component"), true);
      });
    });

    scene.components.forEach(function (node) {
      [node.group, node.modeStar, node.bar.group].forEach(function (target) {
        target.addEventListener("click", function () {
          pick(node.id, false);
        });
      });
    });

    root.classList.add("is-interactive");
    render();
  }

  function componentById(id) {
    return COMPONENTS.filter(function (item) {
      return item.id === id;
    })[0];
  }

  function selectedPoint(rule, winnerId) {
    if (rule === "argmax") {
      var winner = componentById(winnerId);
      return { x: winner.x, y: winner.y };
    }
    return COMPONENTS.reduce(
      function (sum, item) {
        return { x: sum.x + item.weight * item.x, y: sum.y + item.weight * item.y };
      },
      { x: 0, y: 0 }
    );
  }

  /* Builds the action-space and likelihood-chart SVG once; render() only toggles
   * classes and moves the selected marker. */
  function buildScene(host) {
    var svg = el("svg", {
      viewBox: "0 0 420 404",
      class: "gmm-svg gmm-svg-space",
      role: "img",
      "aria-label":
        "Robot-arm action space holding three Gaussian action modes, a candidate action, " +
        "and the action selected for execution."
    });
    var chart = el("svg", {
      viewBox: "0 0 270 340",
      class: "gmm-svg gmm-svg-chart",
      role: "img",
      "aria-label": "Bar chart of the mixture weight of each action mode."
    });

    svg.appendChild(
      el("rect", {
        class: "gmm-space",
        x: SPACE.x,
        y: SPACE.y,
        width: SPACE.w,
        height: SPACE.h,
        rx: 26
      })
    );
    svg.appendChild(text("Robot-arm action space", SPACE.x + SPACE.w / 2, 22, "gmm-heading"));
    chart.appendChild(text("Mode likelihood  \u03c0_k", 135, 22, "gmm-heading"));
    chart.appendChild(
      el("line", { class: "gmm-axis", x1: 22, y1: BARS.base, x2: 248, y2: BARS.base })
    );

    var maxWeight = COMPONENTS.reduce(function (max, item) {
      return Math.max(max, item.weight);
    }, 0);
    var scale = (BARS.base - BARS.top) / maxWeight;

    var components = COMPONENTS.map(function (item, index) {
      /* --- density blob in the action space --- */
      var group = el("g", { class: "gmm-mode" });
      RING_FACTORS.forEach(function (factor, ring) {
        group.appendChild(
          el("ellipse", {
            class: ring === 0 ? "gmm-halo" : "gmm-ring",
            cx: item.x,
            cy: item.y,
            rx: item.rx * factor,
            ry: item.ry * factor,
            fill: item.color,
            "fill-opacity": ring === 0 ? 0.1 : 0.16 + ring * 0.16
          })
        );
      });
      group.appendChild(el("title", {}, item.name + " (\u03c0 = " + item.weight.toFixed(2) + ")"));

      /* Kept outside the blob group so it stays fully opaque and on top when a
       * neighbouring halo overlaps it. */
      var modeStar = star(item.x, item.y, 8.5, "gmm-star gmm-star-mode");
      modeStar.appendChild(el("title", {}, item.name));

      /* --- matching bar in the likelihood chart --- */
      var height = item.weight * scale;
      var bx = BARS.x + index * (BARS.width + BARS.gap);
      var barGroup = el("g", { class: "gmm-bar" });
      barGroup.appendChild(
        el("rect", {
          class: "gmm-bar-fill",
          x: bx,
          y: BARS.base - height,
          width: BARS.width,
          height: height,
          fill: item.color
        })
      );
      barGroup.appendChild(
        text(item.weight.toFixed(2), bx + BARS.width / 2, BARS.base - height - 9, "gmm-bar-value")
      );
      barGroup.appendChild(
        text("\u03c0" + subscript(index + 1), bx + BARS.width / 2, BARS.base + 19, "gmm-bar-label")
      );
      barGroup.appendChild(el("title", {}, item.name));

      return { id: item.id, group: group, modeStar: modeStar, bar: { group: barGroup } };
    });

    components.forEach(function (node) {
      svg.appendChild(node.group);
    });
    components.forEach(function (node) {
      svg.appendChild(node.modeStar);
      chart.appendChild(node.bar.group);
    });

    /* gap marker shown when the mixture expectation is used */
    var gap = el("g", { class: "gmm-gap" });
    var centroid = selectedPoint("expectation", null);
    gap.appendChild(el("circle", { class: "gmm-gap-ring", cx: centroid.x, cy: centroid.y, r: 30 }));
    var gapLabel = el("text", { class: "gmm-gap-label", x: centroid.x, y: centroid.y - 51 });
    gapLabel.appendChild(el("tspan", { x: centroid.x }, "averaged action"));
    gapLabel.appendChild(el("tspan", { x: centroid.x, dy: 15 }, "falls between modes"));
    gap.appendChild(gapLabel);
    svg.appendChild(gap);

    var trail = el("path", { class: "gmm-trail" });
    svg.appendChild(trail);

    var candidate = star(CANDIDATE.x, CANDIDATE.y, 10, "gmm-star gmm-star-candidate");
    svg.appendChild(candidate);

    var selected = star(0, 0, 13, "gmm-star gmm-star-selected");
    svg.appendChild(selected);

    /* legend */
    var legend = el("g", { class: "gmm-legend" });
    [
      ["gmm-star-candidate", "GSA-guided candidate"],
      ["gmm-star-mode", "candidate action modes \u03bc_k"],
      ["gmm-star-selected", "selected executable action \u03c4*"]
    ].forEach(function (entry, index) {
      var ly = 352 + index * 20;
      legend.appendChild(star(18, ly - 4, 7, "gmm-star is-visible " + entry[0]));
      legend.appendChild(text(entry[1], 34, ly, "gmm-legend-label"));
    });
    svg.appendChild(legend);

    var canvas = document.createElement("div");
    canvas.className = "gmm-canvas";
    canvas.appendChild(svg);
    canvas.appendChild(chart);
    host.innerHTML = "";
    host.appendChild(canvas);

    return {
      components: components,
      candidate: candidate,
      selected: selected,
      trail: trail,
      gap: gap
    };
  }

  /* ==================================================================
   * helpers
   * ================================================================== */

  function toArray(list) {
    return Array.prototype.slice.call(list);
  }

  /* Arrow / Home / End handling shared by the two radio groups. */
  function radioKeyTarget(key, index, length) {
    if (key === "ArrowRight" || key === "ArrowDown") {
      return (index + 1) % length;
    }
    if (key === "ArrowLeft" || key === "ArrowUp") {
      return (index - 1 + length) % length;
    }
    if (key === "Home") {
      return 0;
    }
    if (key === "End") {
      return length - 1;
    }
    return null;
  }

  function el(name, attrs, textContent) {
    var node = document.createElementNS(SVG_NS, name);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, String(attrs[key]));
    });
    if (textContent) {
      node.textContent = textContent;
    }
    return node;
  }

  function text(value, x, y, className) {
    return el("text", { x: x, y: y, class: className }, value);
  }

  function star(cx, cy, radius, className) {
    var points = [];
    for (var i = 0; i < 10; i += 1) {
      var r = i % 2 === 0 ? radius : radius * 0.42;
      var angle = (Math.PI / 5) * i - Math.PI / 2;
      points.push((r * Math.cos(angle)).toFixed(2) + "," + (r * Math.sin(angle)).toFixed(2));
    }
    var group = el("g", { class: className, transform: "translate(" + cx + "," + cy + ")" });
    group.appendChild(el("polygon", { points: points.join(" ") }));
    return group;
  }

  function subscript(value) {
    var glyphs = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089";
    return String(value)
      .split("")
      .map(function (digit) {
        return glyphs.charAt(Number(digit));
      })
      .join("");
  }

  /* ==================================================================
   * 3. Semantic anchor bank (paper Section 3.1 and Figure 4)
   * ================================================================== */

  /* Transcribed verbatim from Real_robot/semantic_anchors.json. */
  var ANCHORS = {
    joyful: {
      style_descriptor: "joyful melody",
      technical_annotation: "allegro moderato, 4/4 time",
      tempo_range: [120, 140],
      robot_playability_tags: ["short motif", "clear onset", "reachable bells"]
    },
    energetic: {
      style_descriptor: "energetic rhythm",
      technical_annotation: "vivace, syncopated rhythm",
      tempo_range: [140, 180],
      robot_playability_tags: ["limited leaps", "strong onsets", "stable tempo"]
    },
    peaceful: {
      style_descriptor: "peaceful harmony",
      technical_annotation: "andante, legato phrasing",
      tempo_range: [60, 80],
      robot_playability_tags: ["slow tempo", "gentle strike", "small interval"]
    },
    dramatic: {
      style_descriptor: "dramatic expression",
      technical_annotation: "forte, staccato expression",
      tempo_range: [100, 120],
      robot_playability_tags: ["accented strike", "clear rhythm", "safe reach"]
    }
  };

  /* The motif printed in the paper's prompt figure (C4 E4 G4), written as MIDI
   * numbers because create_music_prompt() formats seed notes that way. */
  var SEED_NOTES = "Note60 | Note64 | Note67";

  /* Both responses are the two printed in the paper's prompt figure for the same
   * joyful anchor, motif, and observed instrument state. Nothing here is inferred
   * for other moods, because the paper works through this one case only. */
  var MODES = {
    replay: {
      laneSub: "repeats",
      /* A playback robot is handed the notes, so the observed instrument state
         never enters the decision. The bell row dims to say so. */
      bellsIdle: true,
      bellsSub: "camera ignored",
      foot:
        "The three blocks land on the human's blocks exactly, and nothing about the "
        + "instrument had to be understood to produce them.",
      verdict: [
        { ok: false, text: "echoes the input" },
        { ok: false, text: "no reasoning about which bells are reachable" }
      ]
    },
    anchored: {
      laneSub: "answers",
      bellsIdle: false,
      bellsSub: "camera checked",
      foot:
        "The plan never reaches for F, so the arm keeps clear of the blocked bell.",
      verdict: [
        { ok: true, text: "new material, not a copy" },
        { ok: true, text: "avoids the occluded bell" }
      ]
    }
  };

  function initAnchorExplorer() {
    var root = document.querySelector("[data-anchor-explorer]");
    if (!root) {
      return;
    }

    var chips = toArray(root.querySelectorAll("[data-anchor]"));
    var prompt = root.querySelector("[data-anchor-prompt]");
    var fields = {};
    toArray(root.querySelectorAll("[data-anchor-field]")).forEach(function (node) {
      fields[node.getAttribute("data-anchor-field")] = node;
    });

    /* Line for line the string returned by create_music_prompt(). Only the two
     * flagged lines are written by the anchor. */
    function promptLines(mood, anchor) {
      return [
        { text: "You are a musical co-creator for a physical chime-playing robot." },
        { text: "Generate a complementary response rather than copying the human seed." },
        {
          text:
            "Return JSON with fields: intent, style, tempo, human_seed, robot_role, " +
            "available_notes, and robot_notes."
        },
        { text: "" },
        {
          text:
            "Style anchor: " +
            anchor.style_descriptor +
            " - " +
            anchor.technical_annotation,
          fromAnchor: true
        },
        {
          text:
            "Robot playability constraints: " +
            anchor.robot_playability_tags.join(", "),
          fromAnchor: true
        },
        { text: "Current human seed notes: [" + SEED_NOTES + "]" },
        { text: "User command: play something " + mood + " with me" }
      ];
    }

    function select(mood, moveFocus) {
      var anchor = ANCHORS[mood];
      if (!anchor) {
        return;
      }

      chips.forEach(function (chip) {
        var on = chip.getAttribute("data-anchor") === mood;
        chip.setAttribute("aria-checked", on ? "true" : "false");
        chip.tabIndex = on ? 0 : -1;
        if (on && moveFocus) {
          chip.focus();
        }
      });

      fields.style_descriptor.textContent = anchor.style_descriptor;
      fields.technical_annotation.textContent = anchor.technical_annotation;
      fields.tempo_range.textContent =
        anchor.tempo_range[0] + " \u2013 " + anchor.tempo_range[1] + " BPM";

      fields.tags.textContent = "";
      anchor.robot_playability_tags.forEach(function (tag) {
        var node = document.createElement("span");
        node.className = "anchor-tag";
        node.textContent = tag;
        fields.tags.appendChild(node);
      });

      /* Real newline nodes rather than block-level spans, so selecting the block
       * copies the prompt with its line breaks intact. */
      prompt.textContent = "";
      promptLines(mood, anchor).forEach(function (line, index) {
        if (index > 0) {
          prompt.appendChild(document.createTextNode("\n"));
        }
        if (!line.fromAnchor) {
          prompt.appendChild(document.createTextNode(line.text));
          return;
        }
        var node = document.createElement("span");
        node.className = "anchor-line is-anchor";
        node.textContent = line.text;
        prompt.appendChild(node);
      });
    }

    chips.forEach(function (chip, index) {
      chip.addEventListener("click", function () {
        select(chip.getAttribute("data-anchor"), false);
      });
      chip.addEventListener("keydown", function (event) {
        var next = radioKeyTarget(event.key, index, chips.length);
        if (next === null) {
          return;
        }
        event.preventDefault();
        select(chips[next].getAttribute("data-anchor"), true);
      });
    });

    /* ---- replay versus anchored planning ---- */

    var modeButtons = toArray(root.querySelectorAll("[data-mode]"));
    var tracks = toArray(root.querySelectorAll("[data-track]"));
    var bells = root.querySelector("[data-bells]");
    var bellsSub = root.querySelector("[data-bells-sub]");
    var laneSub = root.querySelector("[data-lane-sub]");
    var stageFoot = root.querySelector("[data-stage-foot]");
    var outVerdict = root.querySelector("[data-outcome-verdict]");

    function selectMode(mode, moveFocus) {
      var spec = MODES[mode];
      if (!spec) {
        return;
      }

      modeButtons.forEach(function (button) {
        var on = button.getAttribute("data-mode") === mode;
        button.setAttribute("aria-checked", on ? "true" : "false");
        button.tabIndex = on ? 0 : -1;
        if (on && moveFocus) {
          button.focus();
        }
      });

      /* Only one response track is drawn at a time; both occupy the same lane. */
      tracks.forEach(function (track) {
        var on = track.getAttribute("data-track") === mode;
        track.style.display = on ? "" : "none";
      });

      bells.classList.toggle("is-idle", spec.bellsIdle);
      bellsSub.textContent = spec.bellsSub;
      laneSub.textContent = spec.laneSub;
      stageFoot.textContent = spec.foot;

      outVerdict.textContent = "";
      spec.verdict.forEach(function (item) {
        var li = document.createElement("li");
        li.className = item.ok ? "is-yes" : "is-no";
        li.textContent = item.text;
        outVerdict.appendChild(li);
      });
    }

    modeButtons.forEach(function (button, index) {
      button.addEventListener("click", function () {
        selectMode(button.getAttribute("data-mode"), false);
      });
      button.addEventListener("keydown", function (event) {
        var next = radioKeyTarget(event.key, index, modeButtons.length);
        if (next === null) {
          return;
        }
        event.preventDefault();
        selectMode(modeButtons[next].getAttribute("data-mode"), true);
      });
    });

    root.classList.add("is-interactive");
    select("joyful", false);
    selectMode("anchored", false);
  }

  function boot() {
    initGsaExplorer();
    initGmmExplorer();
    initAnchorExplorer();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
