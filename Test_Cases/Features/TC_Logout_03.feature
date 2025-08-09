Feature: ZARA Logout Simulation

  Scenario: Simulate logout by navigating to homepage
    Given the user is logged into ZARA and has rejected cookies
    When the user clicks on their profile link
    When the user clicks the logout button
    Then the user should be redirected to the ZARA homepage
