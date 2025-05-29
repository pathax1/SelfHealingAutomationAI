Feature: ZARA Logout Simulation

  Scenario: Simulate logout by navigating to homepage
    Given the user is on the ZARA login page
    When the user clicks on Logout
    Then the user should be redirected to the ZARA homepage
